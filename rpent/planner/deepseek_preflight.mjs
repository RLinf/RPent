// Copyright 2026 The RPent Authors.
// Licensed under the Apache License, Version 2.0.

/** Local catalog and credential metadata only: never starts an agent or request. */
import { renameSync, writeFileSync } from 'node:fs';

export const inject = ['llm', 'credentials'];

export async function apply(ctx, config) {
  ctx.on('llm/stream', async function* () {
    throw new Error('Model requests are disabled during RPent configuration preflight');
  });
  const report = (value) => {
    const temporary = `${config.resultPath}.tmp`;
    writeFileSync(temporary, JSON.stringify(value), { mode: 0o600 });
    renameSync(temporary, config.resultPath);
  };
  let phase = 'catalog';
  try {
    const info = await ctx.llm.resolveModelInfo(config.provider, config.model);
    const efforts = (info.reasoning?.efforts ?? []).map(effort => effort.id);
    const modalities = info.inputModalities ?? [];
    if (!efforts.includes(config.reasoningEffort)
        || (config.requireImages && !modalities.includes('image'))) {
      report({ status: 'unsupported_capability' });
      return;
    }
    // listRecords exposes key/kind only; a grant remains opaque in its store.
    phase = 'credentials';
    const records = await ctx.credentials.listRecords();
    const credential = records.find(record => record.key === `llm-pi-ai/${config.provider}`);
    if (credential?.kind !== 'grant') {
      report({ status: 'missing_credential' });
      return;
    }
    report({
      status: 'ready', provider: config.provider, model: config.model,
      reasoning_effort: config.reasoningEffort,
      reasoning_efforts: efforts, input_modalities: modalities,
      credential_configured: true, credential_kind: credential.kind,
    });
  } catch {
    // Exception bodies may contain provider diagnostics. Do not export them.
    report({ status: 'inspection_failed', phase });
  }
}
