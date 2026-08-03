export function createTaskSuiteSuggester() {
  let command = "";
  let suites = [];

  function configure(config) {
    command = typeof config?.command === "string" ? config.command : "";
    suites = Array.isArray(config?.suites)
      ? [...new Set(config.suites.filter(value => typeof value === "string"))]
      : [];
  }

  function suiteContext(value, selectionStart, selectionEnd) {
    if (
      !command
      || !suites.length
      || value.includes("\n")
      || selectionStart !== selectionEnd
      || selectionEnd !== value.length
      || !value.startsWith(command)
    ) return null;

    const match = value.slice(command.length).match(/^[\t ]+([^\t ]*)$/);
    if (!match) return null;
    return {
      leading: value.slice(0, value.length - match[1].length),
      prefix: match[1],
    };
  }

  function suggest(value, selectionStart, selectionEnd) {
    const context = suiteContext(value, selectionStart, selectionEnd);
    if (!context) return [];
    return suites.filter(suite => suite.startsWith(context.prefix));
  }

  function select(value, selectionStart, selectionEnd, suite) {
    const context = suiteContext(value, selectionStart, selectionEnd);
    if (!context || !suites.includes(suite) || !suite.startsWith(context.prefix)) {
      return null;
    }
    const selected = `${context.leading}${suite} `;
    return { value: selected, cursor: selected.length };
  }

  return { configure, select, suggest };
}
