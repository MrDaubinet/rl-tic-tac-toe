import path from 'path';
import resolve from '@rollup/plugin-node-resolve';
import replace from '@rollup/plugin-replace';
import commonjs from '@rollup/plugin-commonjs';
import svelte from 'rollup-plugin-svelte';
import babel from '@rollup/plugin-babel';
import { terser } from 'rollup-plugin-terser';
import sveltePreprocess from "svelte-preprocess";
import typescript from '@rollup/plugin-typescript';
import config from 'sapper/config/rollup.js';
import pkg from './package.json';
import url from '@rollup/plugin-url';

const mode = process.env.NODE_ENV;
const dev = mode === 'development';
const legacy = !!process.env.SAPPER_LEGACY_BUILD;

const onwarn = (warning, onwarn) =>
	(warning.code === 'MISSING_EXPORT' && /'preload'/.test(warning.message)) ||
	(warning.code === 'CIRCULAR_DEPENDENCY' && /[/\\]@sapper[/\\]/.test(warning.message)) ||
	(warning.code === 'THIS_IS_UNDEFINED') ||
	onwarn(warning);

export default {
  client: {
    input: config.client.input().replace(/\.js$/, '.ts'),
    output: config.client.output(),
    preserveEntrySignatures: false,
    plugins: [
      // Step 1 -> Inject Node environment variables
      replace({
        'process.browser': true,
        'process.env.NODE_ENV': JSON.stringify(mode)
      }),
      // Step 2 -> Compile Svelte
      svelte({
        preprocess: sveltePreprocess(),
				compilerOptions: {
          dev,
					hydratable: true
				},
        emitCss: true,
      }),
      // Step 3 -> imports files as data-URIs or ES Modules.
      url({
				sourceDir: path.resolve(__dirname, 'src/node_modules/images'),
				publicPath: '/client/'
			}),
      // Step 4 -> Import external dependancies from node moduels
      resolve({
        browser: true,
        dedupe: ['svelte']
      }),
      // Step 5 -> Make all require imports ES6 compatible
      commonjs(),
			typescript({ sourceMap: dev }),
      // If legacy flag
      legacy &&
      // Step 7 -> Make Everything backwacks js compatible	
      babel({
        extensions: ['.js', '.mjs', '.html', '.svelte'],
        babelHelpers: 'runtime',
        exclude: ['node_modules/@babel/**'],
        presets: [
          ['@babel/preset-env', {
            targets: '> 0.25%, not dead'
          }]
        ],
        plugins: [
          '@babel/plugin-syntax-dynamic-import',
          ['@babel/plugin-transform-runtime', {
            useESModules: true
          }]
        ]
      }),

      !dev && terser({
        module: true
      })
    ],

    preserveEntrySignatures: false,
    onwarn,
  },

  server: {
    input: { server: config.server.input().server.replace(/\.js$/, ".ts") },
    output: config.server.output(),
    plugins: [
      replace({
        'process.browser': false,
        'process.env.NODE_ENV': JSON.stringify(mode)
      }),
      svelte({
        preprocess: sveltePreprocess(),
        compilerOptions: {
          generate: 'ssr',
          dev,
        }
      }),
      // postcss({ extract: "bundle.css" }),
      resolve({
        dedupe: ['svelte']
      }),
      commonjs(),
			typescript({ sourceMap: dev })
    ],
    external: Object.keys(pkg.dependencies).concat(
      require('module').builtinModules || Object.keys(process.binding('natives'))
    ),

    preserveEntrySignatures: 'strict',
    onwarn,
  },

  serviceworker: {
    input: config.serviceworker.input().replace(/\.js$/, '.ts'),
    output: config.serviceworker.output(),
    plugins: [
      resolve(),
      replace({
        'process.browser': true,
        'process.env.NODE_ENV': JSON.stringify(mode)
      }),
      commonjs(),
			typescript({ sourceMap: dev }),
      !dev && terser()
    ],

    preserveEntrySignatures: false,
    onwarn,
  }
};
