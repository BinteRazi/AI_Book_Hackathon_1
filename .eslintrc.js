module.exports = {
  root: true,
  extends: [
    '@docusaurus',
    'prettier',
  ],
  plugins: ['prettier'],
  rules: {
    'prettier/prettier': 'error',
  },
  overrides: [
    {
      files: ['*.ts', '*.tsx'],
      parser: '@typescript-eslint/parser',
      parserOptions: {
        project: './tsconfig.json',
      },
      extends: [
        '@docusaurus',
        'prettier',
      ],
    },
  ],
};