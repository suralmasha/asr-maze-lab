import js from "@eslint/js";
import globals from "globals";

export default [
  {
    ignores: [".agents/**", ".venv/**", "node_modules/**"],
  },
  {
    ...js.configs.recommended,
    files: ["web/*.js"],
    languageOptions: {
      sourceType: "script",
      globals: {
        ...globals.browser,
        status: "off",
        labConfig: "readonly",
      },
    },
    rules: {
      ...js.configs.recommended.rules,
      "no-unused-vars": ["error", { vars: "local" }],
    },
  },
  {
    files: ["web/*.js"],
    ignores: ["web/recorder.js"],
    languageOptions: {
      globals: { VoiceRecorder: "readonly" },
    },
  },
  {
    files: ["web/*.js"],
    ignores: ["web/audio-settings.js"],
    languageOptions: {
      globals: { audioSettings: "readonly" },
    },
  },
  {
    files: ["web/*.js"],
    ignores: ["web/configure.js"],
    languageOptions: {
      globals: { labDefaults: "readonly" },
    },
  },
  {
    ...js.configs.recommended,
    files: ["config/*.mjs"],
  },
];
