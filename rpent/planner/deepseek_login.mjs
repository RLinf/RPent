// Run a configuration-time sign-in through the runtime's own credential store.
// No agent session or model request is created by this plugin.
export const inject = ['authorization'];

export function apply(ctx) {
  const emit = (event) => process.stderr.write('RPENT_DSH_LOGIN ' + JSON.stringify(event) + '\n');
  void (async () => {
    const key = 'llm-pi-ai/openai-codex';
    try {
      const deadline = Date.now() + 30000;
      while (!ctx.authorization.describe(key)) {
        if (Date.now() >= deadline) throw new Error('Authorization flow unavailable');
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      const result = await ctx.authorization.begin({
        key,
        method: 'oauth',
        interaction: {
          notify(notice) {
            if (notice.url && notice.code) {
              emit({event: 'device_code', url: notice.url, code: notice.code});
            }
          },
          async prompt(prompt) {
            if (prompt.kind === 'select' && prompt.options?.some(option => option.id === 'device_code')) {
              return 'device_code';
            }
            throw new Error('Device-code login is unavailable');
          },
        },
      });
      emit({event: 'settled', status: result.status});
    } catch (error) {
      // Provider exceptions can contain response bodies. Keep credentials out
      // of CLI output by reporting only the error class and stable code.
      emit({event: 'error', name: error?.name || 'Error', code: error?.code || 'LOGIN_FAILED'});
    }
  })();
}
