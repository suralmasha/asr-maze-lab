"use strict";

const labDefaults = {
  ...labConfig.defaults,
  samples_per_class: labConfig.audio.max_samples,
  hop_length: labConfig.defaults.hop_length ?? labConfig.defaults.n_fft / 4,
};

for (const element of document.querySelectorAll("[data-config]")) {
  const value = element.dataset.config
    .split(".")
    .reduce((config, key) => config[key], labConfig);
  element.textContent = Number(value).toLocaleString("ru-RU");
}

for (const attribute of ["min", "max", "step", "value", "checked"]) {
  for (const element of document.querySelectorAll(
    `[data-config-${attribute}]`,
  )) {
    const value = element
      .getAttribute(`data-config-${attribute}`)
      .split(".")
      .reduce((config, key) => config[key], labConfig);
    element[attribute] = value;
  }
}
