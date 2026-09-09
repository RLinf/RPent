const SCALAR_TYPES = new Set(["string", "number", "integer", "boolean"]);

function normalizedTypes(schema) {
  const declared = schema?.type;
  if (typeof declared === "string") return [declared];
  if (!Array.isArray(declared)) return [];
  return declared.filter(type => type !== "null");
}

export function analyzePrimitiveSchema(primitive) {
  const schema = primitive?.input_schema;
  if (!schema || schema.type !== "object" || !schema.properties ||
      typeof schema.properties !== "object" || Array.isArray(schema.properties)) {
    return { supported: false, reason: "input_schema must be an object with properties" };
  }
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  const fields = [];
  for (const [name, fieldSchema] of Object.entries(schema.properties)) {
    const types = normalizedTypes(fieldSchema);
    if (types.length === 1 && SCALAR_TYPES.has(types[0])) {
      fields.push({
        name,
        type: types[0],
        schema: fieldSchema,
        required: required.has(name),
      });
      continue;
    }
    if (types.length === 2 && types.includes("number") && types.includes("string")) {
      fields.push({
        name,
        type: "number-or-string",
        schema: fieldSchema,
        required: required.has(name),
      });
      continue;
    }
    const itemTypes = normalizedTypes(fieldSchema?.items);
    const length = fieldSchema?.minItems;
    if (types.length === 1 && types[0] === "array" &&
        itemTypes.length === 1 && itemTypes[0] === "number" &&
        Number.isInteger(length) && length > 0 && fieldSchema.maxItems === length) {
      fields.push({
        name,
        type: "number-array",
        length,
        schema: fieldSchema,
        required: required.has(name),
      });
      continue;
    }
    return { supported: false, reason: `unsupported schema for “${name}”` };
  }
  const names = new Set(fields.map(field => field.name));
  if ([...required].some(name => !names.has(name))) {
    return { supported: false, reason: "required field is missing from properties" };
  }
  return { supported: true, fields };
}

function makeInput(field, copy) {
  if (field.type === "boolean") {
    const select = document.createElement("select");
    for (const [value, label] of [
      ["", copy.notSet],
      ["true", "true"],
      ["false", "false"],
    ]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    }
    return select;
  }
  if (field.type === "string" && Array.isArray(field.schema.enum)) {
    const select = document.createElement("select");
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = copy.notSet;
    select.appendChild(blank);
    for (const value of field.schema.enum) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    }
    return select;
  }
  const input = document.createElement("input");
  input.type = ["string", "number-or-string"].includes(field.type)
    ? "text"
    : "number";
  if (field.type === "integer") input.step = "1";
  if (field.type === "number") input.step = "any";
  for (const attribute of ["minimum", "maximum"]) {
    if (Number.isFinite(field.schema[attribute])) {
      input[attribute === "minimum" ? "min" : "max"] = field.schema[attribute];
    }
  }
  return input;
}

export function renderPrimitiveFields(container, fields, copy) {
  const nodes = fields.map(field => {
    const wrapper = document.createElement("div");
    wrapper.className = "primitive-field";
    wrapper.dataset.field = field.name;
    const label = document.createElement("label");
    label.textContent = field.name;
    if (field.required) {
      const marker = document.createElement("span");
      marker.className = "primitive-required";
      marker.textContent = " *";
      label.appendChild(marker);
    }
    wrapper.appendChild(label);

    if (field.type === "number-array") {
      const row = document.createElement("div");
      row.className = "primitive-array";
      for (let index = 0; index < field.length; index++) {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.dataset.arrayIndex = String(index);
        input.setAttribute("aria-label", `${field.name} ${index + 1}`);
        row.appendChild(input);
      }
      wrapper.appendChild(row);
    } else {
      const input = makeInput(field, copy);
      input.setAttribute("aria-label", field.name);
      wrapper.appendChild(input);
    }
    return wrapper;
  });
  container.replaceChildren(...nodes);
}

export function readPrimitiveArguments(form, fields, copy) {
  const argumentsObject = {};
  const wrappers = [...form.querySelectorAll(".primitive-field")];
  for (const field of fields) {
    const wrapper = wrappers.find(node => node.dataset.field === field.name);
    wrapper?.classList.remove("invalid");
    const inputs = [...(wrapper?.querySelectorAll("input, select") || [])];
    const values = inputs.map(input => input.value.trim());
    const empty = values.every(value => value === "");
    if (empty && !field.required) continue;
    if (empty || values.some(value => value === "")) {
      wrapper?.classList.add("invalid");
      inputs.find(input => input.value.trim() === "")?.focus();
      throw new Error(copy.fieldRequired(field.name));
    }
    if (field.type === "string") {
      argumentsObject[field.name] = values[0];
    } else if (field.type === "number-or-string") {
      const number = Number(values[0]);
      argumentsObject[field.name] = Number.isFinite(number) ? number : values[0];
    } else if (field.type === "boolean") {
      argumentsObject[field.name] = values[0] === "true";
    } else if (field.type === "number-array") {
      const numbers = values.map(Number);
      if (numbers.some(value => !Number.isFinite(value))) {
        wrapper?.classList.add("invalid");
        inputs[0]?.focus();
        throw new Error(copy.invalidField(field.name));
      }
      argumentsObject[field.name] = numbers;
    } else {
      const value = Number(values[0]);
      if (!Number.isFinite(value) ||
          (field.type === "integer" && !Number.isInteger(value)) ||
          (Number.isFinite(field.schema.minimum) && value < field.schema.minimum) ||
          (Number.isFinite(field.schema.maximum) && value > field.schema.maximum) ||
          (field.schema.const !== undefined && value !== field.schema.const)) {
        wrapper?.classList.add("invalid");
        inputs[0]?.focus();
        throw new Error(copy.invalidField(field.name));
      }
      argumentsObject[field.name] = value;
    }
  }
  return argumentsObject;
}
