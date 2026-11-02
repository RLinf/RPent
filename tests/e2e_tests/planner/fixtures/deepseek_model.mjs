// Copyright 2026 The RPent Authors.
// Licensed under the Apache License, Version 2.0.

/** Deterministic model inside the real bundled runtime; no network or login. */
import { appendFileSync, writeFileSync } from 'node:fs';

export const name = 'rpent-test-model';
export const inject = ['llm', 'attachments', 'agents', 'compaction'];

function delay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    let timer;
    const stop = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', stop);
      reject(new Error('Test model aborted'));
    };
    if (signal?.aborted) return stop();
    signal?.addEventListener('abort', stop, { once: true });
    if (Number.isFinite(milliseconds)) timer = setTimeout(() => {
      signal?.removeEventListener('abort', stop);
      resolve();
    }, milliseconds);
  });
}

export async function apply(ctx, config) {
  if (config.imageStorageFailure) {
    ctx.attachments.saveImages = async () => { throw new Error('Simulated attachment storage failure'); };
  }
  if (config.imageReadFailure) {
    ctx.attachments.readImageRequest = async () => { throw new Error('Simulated request image read failure'); };
  }
  let index = 0;
  const adapter = {
    providerInfo: provider => ({ id: provider, name: provider }),
    providerRetryPolicy: () => config.retryPolicy ?? undefined,
    imageRequestPricing: () => undefined,
    imageRequestTarget: (_provider, _model, image) => ({
      width: Math.floor(image.width * (config.imageTargetScale ?? 1)),
      height: Math.floor(image.height * (config.imageTargetScale ?? 1)),
      maxBytes: image.bytes,
    }),
    listModels: async () => [],
    resolveModel: async (provider, model) => ({
      provider, id: model, name: model,
      inputModalities: ['text', 'image'],
      context: { contextWindow: config.contextWindow ?? 500000 },
      reasoning: { efforts: [{ id: 'xhigh', name: 'Extra high' }] },
    }),
    async prepareCall(provider, model) {
      return { model: await this.resolveModel(provider, model),
        stream: options => this.stream(options) };
    },
    async *stream(options) {
      appendFileSync(config.requestsPath, JSON.stringify(options) + '\n');
      const entry = config.responses[index++];
      if (!entry) throw new Error('Test model received an unexpected request');
      if (entry.purpose && entry.purpose !== (options.purpose ?? 'root')) {
        throw new Error('Test received a model call with an unexpected purpose');
      }
      if (entry.delayMs) await delay(entry.delayMs, options.signal);
      if (entry.hang) {
        await delay(Infinity, options.signal);
        return;
      }
      if (entry.error) throw new Error(entry.error);
      if (entry.failure) {
        if (entry.usage) yield { type: 'usage', usage: entry.usage };
        yield { type: 'finish', reason: { kind: 'error', failure: entry.failure } };
        return;
      }
      const blocks = entry.calls
        ? entry.calls.map((call, position) => ({ type: 'tool-call',
          id: `call-${index}-${position}`, name: call.name,
          arguments: JSON.stringify(call.arguments ?? {}) }))
        : [{ type: 'text', text: entry.text ?? 'done' }];
      for (let position = 0; position < blocks.length; position++) {
        const block = blocks[position];
        yield { type: 'block-start', index: position, blockType: block.type };
        if (block.type === 'tool-call') {
          yield { type: 'tool-call-delta', index: position,
            id: block.id, name: block.name, argumentsDelta: block.arguments };
        } else {
          yield { type: 'text-delta', index: position, text: block.text };
        }
        yield { type: 'block-end', index: position, block };
      }
      yield { type: 'usage', usage: entry.usage ?? { inputTokens: 12, outputTokens: 7, totalTokens: 19 } };
      yield { type: 'finish', reason: { kind: entry.calls ? 'tool-calls' : 'stop' } };
    },
  };
  ctx.llm.registerAdapter(['openai-codex'], adapter);
  if (config.compactAfterTurn) {
    let requested = false;
    ctx.on('session/event', (session, event) => {
      if (requested || event.type !== 'turn/end' || event.data.turn !== config.compactAfterTurn) return;
      requested = true;
      const agent = ctx.agents.get(session.id);
      void agent.whenIdle().then(async () => {
        try {
          const result = await ctx.compaction.compactNow(agent, new AbortController().signal);
          writeFileSync(config.compactionPath, JSON.stringify({ compacted: result !== null }));
        } catch (error) {
          writeFileSync(config.compactionPath, JSON.stringify({ error: String(error.message ?? error) }));
        }
      });
    });
  }
  ctx.provide('rpentTestModelReady', true);
}
