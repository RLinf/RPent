// Copyright 2026 The RPent Authors.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import { createInteractionController } from './interaction.js';

function setup() {
  const elements = new Map();
  const pending = [];
  const select = id => {
    if (!elements.has(id)) elements.set(id, {
      value: '', dataset: {}, attributes: {}, listeners: {},
      classList: { add() {} },
      replaceChildren() {},
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(name, handler) { this.listeners[name] = handler; },
    });
    return elements.get(id);
  };
  globalThis.fetch = () => new Promise(resolve => pending.push(resolve));
  const copy = {
    composerPlaceholder: 'Message', composerKeys: 'Keys',
    interactionReady: 'READY', interactionBusy: 'BUSY',
    interruptRequested: 'REQUESTED', interruptSucceeded: 'COMPLETED',
    interruptFailed: error => `ERROR ${error}`,
    unknownRequestError: 'Unknown error',
  };
  const controller = createInteractionController({ copy, select });
  const snapshot = (generation = 1, activity = 'busy', requested = false) => ({
    task_generation: generation, session_state: 'running',
    control_error: null, control_feedback: [], operator: null,
    primitives_available: activity === 'idle',
    interaction: {
      input_mode: 'conversation', planner_activity: activity,
      interrupt_requested: requested, messages: [], last_error: null,
    },
  });
  const press = () => select('#chatInput').listeners.keydown({
    key: 'Escape', preventDefault() {},
  });
  const reply = async (index = 0, failed = false) => {
    pending[index]({
      ok: !failed, status: failed ? 500 : 202,
      text: async () => JSON.stringify(failed
        ? { error: 'late failure' } : { interrupt_requested: true }),
    });
    await new Promise(resolve => setImmediate(resolve));
  };
  const status = () => select('#interactionStatus').textContent;
  controller.applySnapshot(snapshot());
  return { controller, select, snapshot, press, reply, status };
}

test('ordinary requested response stays pending until backend ACK', async () => {
  const run = setup();
  try {
    run.press();
    assert.equal(run.status(), 'REQUESTED');
    await run.reply();
    assert.equal(run.status(), 'REQUESTED');
    run.controller.applySnapshot(run.snapshot(1, 'idle'));
    assert.equal(run.status(), 'COMPLETED');
  } finally { run.controller.reset(); }
});

test('backend idle wins while HTTP is pending and after a late requested reply', async () => {
  const run = setup();
  try {
    run.press();
    run.controller.applySnapshot(run.snapshot(1, 'busy', true));
    run.controller.applySnapshot(run.snapshot(1, 'idle'));
    assert.equal(run.status(), 'COMPLETED');
    await run.reply();
    assert.equal(run.status(), 'COMPLETED');
  } finally { run.controller.reset(); }
});

test('coalesced idle ACK also defeats a late HTTP failure', async () => {
  const run = setup();
  try {
    run.press();
    run.controller.applySnapshot(run.snapshot(1, 'idle'));
    assert.equal(run.status(), 'READY');
    await run.reply(0, true);
    assert.equal(run.status(), 'READY');
  } finally { run.controller.reset(); }
});

test('old generation reply cannot mark the new generation interrupted', async () => {
  const run = setup();
  try {
    run.press();
    run.controller.reset();
    run.controller.applySnapshot(run.snapshot(2));
    await run.reply();
    assert.equal(run.status(), 'BUSY');
  } finally { run.controller.reset(); }
});

test('old request completion cannot clear a new request in flight', async () => {
  const run = setup();
  try {
    run.press();
    run.controller.reset();
    run.controller.applySnapshot(run.snapshot(2));
    run.press();
    await run.reply();
    assert.equal(run.select('#chatInput').attributes['aria-busy'], 'true');
    await run.reply(1);
    run.controller.applySnapshot(run.snapshot(2, 'idle'));
    assert.equal(run.status(), 'COMPLETED');
  } finally { run.controller.reset(); }
});
