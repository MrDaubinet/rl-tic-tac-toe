const defaultTheme = require("tailwindcss/defaultTheme");

module.exports = {
  theme: {
    extend: {
      maxWidth: {
        '2xs': '10rem',
        '3xs': '8rem',
        '4xs': '6rem',
        '5xs': '5rem',
        '2xl': '1500',
      }, 
      fontSize: {
        '7xl': '5rem',
        '8xl': '6rem',
        '9xl': '7rem',
        '10xl': '8rem',
      },
      screens: {
        '2xl': '1400px',
        '3xl': '1600px'
      },
      colors: {
        'aus-grey': {
          50: '#F4F4F6',
          100: '#E8EAED',
          200: '#C6CAD3',
          300: '#A4AAB8',
          400: '#5F6A83',
          500: '#1B2A4E',
          600: '#182646',
          700: '#10192F',
          800: '#0C1323',
          900: '#080D17',
        },
        'aus-dark-blue': {
          50: '#F4F7FF',
          100: '#EAEFFF',
          200: '#CAD8FF',
          300: '#A9C0FF',
          400: '#6991FF',
          500: '#2962FF',
          600: '#2558E6',
          700: '#193B99',
          800: '#122C73',
          900: '#0C1D4D',
        },
        'aus-light-blue': {
          50: '#F2FBFE',
          100: '#E6F6FE',
          200: '#C0EAFC',
          300: '#9ADDFB',
          400: '#4FC3F7',
          500: '#03A9F4',
          600: '#0398DC',
          700: '#026592',
          800: '#014C6E',
          900: '#013349',
        }
      },
      variants: {
        animation: ['responsive', 'hover', 'focus'],
      }
    },
  },
  purge: {
    content: ["./src/**/*.svelte", "./src/template.html"],
  },
  plugins: [require("@tailwindcss/ui")],
};
