// Copyright 2026 The RPent Authors.
// Licensed under the Apache License, Version 2.0.

/**
 * Run-local policy for DeepSeek Harness 0.2.1rc2. The Python adapter owns the
 * Toolkit and its authoritative finish result; this plugin enforces the same
 * boundary inside the agent runtime, before another model step or tool starts.
 * It imports no npm packages and can be loaded by the bundled dsh executable.
 */
import { readFileSync, renameSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

export const name = 'rpent-bridge';
export const inject = ['tools', 'systemPrompt', 'agents', 'agentLoop', 'llm', 'attachments'];

/** RPent's MCP adapter emits PNGs; preserve their pixel-coordinate space. */
function pngDimensions(block) {
  if (block.mimeType !== 'image/png' || typeof block.data !== 'string') {
    throw new Error('RPent expected a PNG image from its MCP adapter');
  }
  const bytes = Buffer.from(block.data, 'base64');
  if (bytes.length < 24 || bytes.toString('hex', 0, 8) !== '89504e470d0a1a0a'
      || bytes.toString('ascii', 12, 16) !== 'IHDR') {
    throw new Error('RPent received an invalid PNG image header');
  }
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

function sameDimensions(first, second) {
  return first?.width === second?.width && first?.height === second?.height;
}

function validateConfig(config) {
  if (!Array.isArray(config.toolNames)
      || config.toolNames.some(value => typeof value !== 'string'
        || !/^mcp__rpent__[A-Za-z0-9_-]+$/.test(value))
      || new Set(config.toolNames).size !== config.toolNames.length) {
    throw new TypeError('RPent requires a unique MCP tool allowlist');
  }
  if (typeof config.systemPrompt !== 'string'
      || typeof config.controlPath !== 'string' || !config.controlPath) {
    throw new TypeError('RPent requires systemPrompt and controlPath');
  }
  if (!config.expectedSelection
      || ['provider', 'model', 'reasoningEffort'].some(
        key => typeof config.expectedSelection[key] !== 'string',
      )) {
    throw new TypeError('RPent requires an exact expected model selection');
  }
  if (!Number.isSafeInteger(config.maxSteps) || config.maxSteps < 1) {
    throw new TypeError('RPent maxSteps must be a positive safe integer');
  }
  const timeout = config.startupTimeoutMs ?? 30000;
  if (!Number.isSafeInteger(timeout) || timeout < 1 || timeout > 2147483647) {
    throw new TypeError('RPent startupTimeoutMs must be a positive timer delay');
  }
  return timeout;
}

/** Await the exact expected MCP inventory, without depending on row order. */
async function waitForTools(ctx, names, timeoutMs) {
  const missing = () => names.filter(toolName => !ctx.tools.get(toolName));
  if (missing().length === 0) return;
  await new Promise((resolve, reject) => {
    let timer;
    let dispose = () => {};
    const finish = error => {
      clearTimeout(timer);
      dispose();
      if (error) reject(error);
      else resolve();
    };
    dispose = ctx.on('tools/change', () => {
      if (missing().length === 0) finish();
    });
    timer = setTimeout(() => finish(new Error(
      `RPent MCP tools unavailable: ${missing().join(', ')}`,
    )), timeoutMs);
    // Registration and our first inventory read may occur in the same tick.
    if (missing().length === 0) finish();
  });
}

/** Apply only to a fresh, single-agent SDK runtime owned by one RPent run. */
export async function apply(ctx, config) {
  const startupTimeoutMs = validateConfig(config);
  const allowed = new Set(config.toolNames);
  let rootAgent;
  let activeTools = 0;
  let activityEpoch = 0;
  const ledger = () => {
    if (!config.toolLedgerPath) return undefined;
    try {
      return JSON.parse(readFileSync(config.toolLedgerPath, 'utf8'));
    } catch (error) {
      return error.code === 'ENOENT' ? { seq: 0 } : undefined;
    }
  };
  const state = {
    ready: false,
    steps: 0,
    stopReason: null,
    deniedCalls: [],
    toolNames: [...allowed].sort(),
    selected: null,
    observedSelection: null,
    finishResult: null,
    imageChecks: [],
    quiescent: false,
    activity: 'starting',
    llmRequests: [],
    compactions: [],
  };
  const persist = () => {
    state.activity = state.stopReason ? 'terminal'
      : !rootAgent ? 'starting' : state.quiescent ? 'idle' : 'running';
    state.activeTools = activeTools;
    const temporary = `${config.controlPath}.tmp`;
    writeFileSync(temporary, JSON.stringify(state), { mode: 0o600 });
    renameSync(temporary, config.controlPath);
  };
  const markBusy = () => {
    activityEpoch += 1;
    state.quiescent = false;
    persist();
  };
  const followIdle = () => {
    if (!rootAgent) return;
    const epoch = activityEpoch;
    void rootAgent.whenIdle().then(() => {
      if (epoch === activityEpoch && activeTools === 0) {
        state.quiescent = true;
        persist();
      }
    }).catch(() => {
      state.stopReason ??= 'control_error';
      persist();
    });
  };
  persist();
  try {
    if (ctx.agentLoop.config.maxParallelToolCalls.get() !== 1) {
      throw new Error('RPent requires agent-loop.maxParallelToolCalls = 1');
    }
    ctx.systemPrompt.section({
      name: 'rpent-system-prompt',
      order: 0,
      complete: true,
      interpolate: false,
      text: config.systemPrompt,
    });

    let finished = false;
    const nativeOutcomes = new Map();
    const terminal = () => finished || state.stopReason !== null;
    const rejectImage = (error, toolName) => {
      state.stopReason = 'image_pipeline_error';
      state.failedTool = toolName;
      state.imageError = String(error.message ?? error).slice(0, 500);
      persist();
    };
    const checkImages = async (sourceImages, projectedImages, toolName, signal) => {
      if (projectedImages.length !== sourceImages.length) {
        throw new Error('RPent image projection lost one or more images');
      }
      for (let index = 0; index < sourceImages.length; index++) {
        const source = pngDimensions(sourceImages[index]);
        const ref = projectedImages[index].attachment;
        if (!sameDimensions(source, ref)) {
          throw new Error('RPent image normalization changed the pixel coordinates');
        }
        const { provider, model } = config.expectedSelection;
        const target = ctx.llm.imageRequestTarget(provider, model, ref);
        if (!sameDimensions(source, target)) {
          throw new Error('RPent model image target changed the pixel coordinates');
        }
        const request = await ctx.attachments.readImageRequest(ref, target, signal);
        if (!sameDimensions(source, request)) {
          throw new Error('RPent model image projection changed the pixel coordinates');
        }
        state.imageChecks.push({
          tool: toolName, index, source, attachmentId: ref.attachmentId,
          normalized: { width: ref.width, height: ref.height, bytes: ref.bytes },
          target,
          request: {
            width: request.width, height: request.height,
            bytes: request.data.length, mediaType: request.mediaType,
          },
        });
      }
      persist();
    };
    const checkSelection = (options, observed = false, expected = config.expectedSelection) => {
      const actual = {
        provider: options.provider,
        model: options.model,
        reasoningEffort: options.reasoningEffort,
        maxTokens: options.maxTokens,
      };
      if (observed === 'auxiliary') state.observedSummarySelection = actual;
      else if (observed) state.observedSelection = actual;
      else state.selected = actual;
      if (Object.entries(expected).some(
        ([key, value]) => actual[key] !== value,
      )) {
        state.stopReason = 'model_mismatch';
        persist();
        throw new Error('RPent model selection differs from the requested configuration');
      }
      persist();
    };
    ctx.tools.guard(exec => {
      let reason;
      if (!allowed.has(exec.name)) reason = 'tool is outside the RPent allowlist';
      else if (finished) reason = 'RPent finish has already been accepted';
      else if (state.stopReason === 'budget') reason = 'RPent step budget exhausted';
      else if (state.stopReason === 'tool_transport_error') reason = 'RPent tool outcome is uncertain';
      else if (state.stopReason === 'model_mismatch') reason = 'RPent model selection differs';
      else if (state.stopReason === 'image_pipeline_error') reason = 'RPent image delivery failed';
      else if (state.stopReason) reason = 'RPent run has ended';
      else if (exec.agent !== rootAgent) reason = 'tool is outside the RPent root agent';
      if (reason) {
        state.deniedCalls.push({ name: exec.name, reason });
        persist();
        return reason;
      }
      return undefined;
    });

    // Body success is provisional: rc2 validates, projects and materializes
    // its value after execute() returns. Observe the authoritative final result
    // synchronously so a failed delivery cannot admit a same-batch action.
    ctx.on('tools/result', (exec, result) => {
      const outcome = nativeOutcomes.get(exec.callId);
      if (!outcome) return; // A pre-dispatch denial performed no RPent action.
      nativeOutcomes.delete(exec.callId);
      const code = result.error?.info?.code;
      if (result.isError && (!outcome.knownBusinessError || code === 'INVALID_TOOL_OUTPUT')) {
        state.toolDeliveryErrors ??= [];
        state.toolDeliveryErrors.push({
          callId: exec.callId, name: exec.name,
          code: code ?? 'RESULT_DELIVERY_FAILED',
        });
        // Keep the more specific image diagnosis; a failed finish delivery
        // overrides its success stop reason while preserving finishResult.
        if (state.stopReason !== 'image_pipeline_error') {
          state.stopReason = 'tool_transport_error';
        }
        state.failedTool = exec.name;
        persist();
      }
    });

    // MCP may convert a failed image admission into successful placeholder
    // text. Verify its durable projection before a later same-batch action,
    // including the real provider's request-sized variant. No image bytes are
    // copied into control metadata, and no provider request is made here.
    ctx.on('tools/post-execute', async (exec, result, next) => {
      if (exec.agent !== rootAgent || !allowed.has(exec.name) || result.isError) {
        return next();
      }
      const sourceImages = result.value?.content?.filter(block => block.type === 'image') ?? [];
      if (sourceImages.length === 0) return next();
      try {
        const projectedImages = result.content.filter(block => block.type === 'image');
        await checkImages(sourceImages, projectedImages, exec.name, exec.signal);
      } catch (error) {
        rejectImage(error, exec.name);
        return { kind: 'block', feedback: [{ type: 'text', text: state.imageError }] };
      }
      return next();
    });

    ctx.on('agent/created', ({ agent }) => {
      if (rootAgent !== undefined) {
        throw new Error('RPent does not allow additional agents or session resume');
      }
      rootAgent = agent;
      checkSelection(agent.options);
      agent.ctx.tools.restrict({ allow: [...allowed] });
      agent.ctx.tools.presentAs('native');
      for (const toolName of allowed) {
        const original = ctx.tools.get(toolName);
        if (!original) throw new Error(`RPent MCP tool disappeared: ${toolName}`);
        agent.ctx.tools.register({
          ...original,
          async execute(args, exec) {
            activeTools += 1;
            markBusy();
            const outcome = { knownBusinessError: false };
            nativeOutcomes.set(exec.callId, outcome);
            try {
              const before = ledger();
              let result;
              try {
                result = await original.execute(args, exec);
              } catch (error) {
                const after = ledger();
                const knownBusinessError = before && after
                  && after.seq === before.seq + 1
                  && after.name === toolName.slice('mcp__rpent__'.length)
                  && after.isError === true
                  && after.executionUncertain === false
                  && typeof after.canonicalBusinessErrorSha256 === 'string'
                  && createHash('sha256').update(String(error.message ?? error)).digest('hex')
                    === after.canonicalBusinessErrorSha256;
                outcome.knownBusinessError = Boolean(knownBusinessError);
                if (!knownBusinessError) {
                  state.stopReason = 'tool_transport_error';
                  state.failedTool = toolName;
                  persist();
                }
                throw error;
              }
              const after = ledger();
              if (after?.executionUncertain === true) {
                state.stopReason = 'tool_transport_error';
                state.failedTool = toolName;
                persist();
                throw new Error('RPent tool execution is uncertain; no further action is allowed');
              }
              const data = result?.structuredContent;
              if (data && data._finish === true && !result.isError) {
                // Keep this latch monotonic even if later result delivery fails:
                // the Python Toolkit has already accepted finish at this point.
                finished = true;
                state.finishResult = data;
                state.stopReason = 'finish';
                persist();
                exec.concludeTurn();
              }
              return result;
            } finally {
              activeTools -= 1;
              persist();
              followIdle();
            }
          },
        });
      }
      const visible = agent.ctx.tools.schemas(agent).map(tool => tool.name).sort();
      if (JSON.stringify(visible) !== JSON.stringify(state.toolNames)) {
        throw new Error(`RPent tool inventory differs: ${visible.join(', ')}`);
      }
      state.sessionId = agent.session.id;
      persist();
      followIdle();
    });

    ctx.on('llm/stream', async function* (options, next) {
      markBusy();
      const auxiliary = Boolean(options.purpose);
      const expected = config.expectedSelection;
      // rc2's compaction summarizer omits effort. Supply the explicitly
      // configured effort before model resolution; never rewrite model/provider.
      if (options.purpose === 'compaction' && options.reasoningEffort === undefined) {
        options.reasoningEffort = expected.reasoningEffort;
      }
      const request = {
        id: state.llmRequests.length + 1,
        purpose: options.purpose ?? 'root',
        selection: {
          provider: options.provider, model: options.model,
          reasoningEffort: options.reasoningEffort, maxTokens: options.maxTokens,
        },
        usage: null, finished: false, finishReason: null, error: null,
        // One middleware invocation is observable. A provider can retry below
        // this seam, so do not manufacture an exact HTTP-attempt count.
        providerAttempts: null,
      };
      state.llmRequests.push(request);
      try {
        if (!rootAgent || options.sessionId !== rootAgent.session.id) {
          state.stopReason ??= 'model_mismatch';
          throw new Error('RPent model request is outside its root session');
        }
        if (terminal()) throw new Error('RPent terminal run cannot make another model request');
        checkSelection(options, auxiliary ? 'auxiliary' : true, expected);
        for await (const chunk of next()) {
          if (chunk.type === 'usage') request.usage = { ...chunk.usage };
          if (chunk.type === 'finish') {
            request.finished = true;
            request.finishReason = chunk.reason.kind;
            if (['error', 'aborted'].includes(chunk.reason.kind)) {
              request.error = { code: chunk.reason.failure?.code ?? chunk.reason.kind };
              if (options.purpose === 'compaction') state.stopReason ??= 'compaction_error';
            }
          }
          yield chunk;
        }
      } catch (error) {
        request.error = { code: typeof error.code === 'string' ? error.code : 'STREAM_ERROR' };
        if (options.purpose === 'compaction') state.stopReason ??= 'compaction_error';
        throw error;
      } finally {
        persist();
        followIdle();
      }
    });

    ctx.on('agent/pre-step', async ({ agent }, next) => {
      if (agent !== rootAgent) throw new Error('RPent received a foreign agent step');
      if (terminal()) {
        return { kind: 'reject' };
      }
      if (state.steps >= config.maxSteps) {
        state.stopReason = 'budget';
        persist();
        return { kind: 'reject' };
      }
      const decision = await next();
      if (decision.kind === 'enter') {
        state.steps += 1;
        markBusy();
      }
      return decision;
    });

    ctx.on('agent/status', ({ agent, status }) => {
      if (agent !== rootAgent) return;
      if (status === 'running') markBusy();
      else followIdle();
    });
    ctx.on('session/event', (session, event) => {
      if (session !== rootAgent?.session) return;
      if (event.type === 'turn/end') {
        followIdle();
      } else if (event.type.startsWith('compaction/')) {
        if (event.type === 'compaction/start') markBusy();
        if (event.type === 'compaction/end' && event.data.error) {
          state.stopReason ??= 'compaction_error';
        }
        // Retain provenance and metering, never summary text or replay state.
        state.compactions.push({
          type: event.type, seq: event.seq,
          ...(event.data.provider ? { provider: event.data.provider } : {}),
          ...(event.data.model ? { model: event.data.model } : {}),
          ...(event.data.usage ? { usage: event.data.usage } : {}),
          ...(event.data.error ? { failed: true } : {}),
        });
        persist();
        if (event.type === 'compaction/end') followIdle();
      }
    });

    await waitForTools(ctx, [...allowed], startupTimeoutMs);
    state.ready = true;
    persist();
    // sdk-jsonrpc-server must inject this service in addition to its defaults.
    // Optional plugin failure can therefore never produce a usable SDK runtime.
    ctx.provide('rpentBridgeReady', { ready: true });
  } catch (error) {
    state.startupError = String(error);
    persist();
    throw error;
  }
}
