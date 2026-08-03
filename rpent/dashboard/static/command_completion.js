function commonPrefix(values) {
  if (!values.length) return "";
  let prefix = values[0];
  for (const value of values.slice(1)) {
    while (prefix && !value.startsWith(prefix)) {
      prefix = prefix.slice(0, -1);
    }
    if (!prefix) break;
  }
  return prefix;
}

export function createTaskCommandCompleter() {
  let command = "";
  let suites = [];
  let cycle = null;

  function reset() {
    cycle = null;
  }

  function configure(config) {
    command = typeof config?.command === "string" ? config.command : "";
    suites = Array.isArray(config?.suites)
      ? [...new Set(config.suites.filter(value => typeof value === "string"))]
      : [];
    reset();
  }

  function result(value) {
    return { value, cursor: value.length };
  }

  function complete(value, selectionStart, selectionEnd) {
    if (
      !command
      || !suites.length
      || !value.startsWith("/")
      || value.includes("\n")
      || selectionStart !== selectionEnd
      || selectionEnd !== value.length
    ) {
      reset();
      return null;
    }

    if (cycle?.value === value) {
      cycle.index = (cycle.index + 1) % cycle.matches.length;
      cycle.value = `${cycle.leading}${cycle.matches[cycle.index]}`;
      return result(cycle.value);
    }
    reset();

    if (!/[\t ]/.test(value)) {
      return command.startsWith(value) ? result(`${command} `) : null;
    }

    if (!value.startsWith(command)) return null;
    const remainder = value.slice(command.length);
    const match = remainder.match(/^[\t ]+([^\t ]*)$/);
    if (!match) return null;

    const suitePrefix = match[1];
    const matches = suites.filter(suite => suite.startsWith(suitePrefix));
    if (!matches.length) return null;

    const leading = value.slice(0, value.length - suitePrefix.length);
    if (suites.includes(suitePrefix)) return result(`${leading}${suitePrefix} `);
    if (matches.length === 1) return result(`${leading}${matches[0]} `);

    const shared = commonPrefix(matches);
    if (shared.length > suitePrefix.length) return result(`${leading}${shared}`);

    cycle = {
      leading,
      matches,
      index: 0,
      value: `${leading}${matches[0]}`,
    };
    return result(cycle.value);
  }

  return { complete, configure, reset };
}
