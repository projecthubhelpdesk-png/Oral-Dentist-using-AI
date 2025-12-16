/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  safelist: [
    // Teal theme
    'from-teal-400', 'via-teal-500', 'to-teal-600', 'bg-teal-500', 'hover:bg-teal-600',
    'text-teal-500', 'text-teal-600', 'text-teal-800', 'border-teal-100', 'border-teal-500',
    'from-teal-50', 'bg-teal-50', 'focus:ring-teal-500',
    // Blue theme
    'from-blue-400', 'via-blue-500', 'to-indigo-600', 'bg-blue-500', 'hover:bg-blue-600',
    'text-blue-500', 'text-blue-600', 'text-blue-800', 'border-blue-100', 'border-blue-500',
    'from-blue-50', 'bg-blue-50', 'focus:ring-blue-500',
    // Emerald theme
    'from-emerald-400', 'via-emerald-500', 'to-green-600', 'bg-emerald-500', 'hover:bg-emerald-600',
    'text-emerald-500', 'text-emerald-600', 'text-emerald-800', 'border-emerald-100', 'border-emerald-500',
    'from-emerald-50', 'bg-emerald-50', 'focus:ring-emerald-500',
    // Purple theme
    'from-purple-400', 'via-purple-500', 'to-indigo-600', 'bg-purple-500', 'hover:bg-purple-600',
    'text-purple-500', 'text-purple-600', 'text-purple-800', 'border-purple-100', 'border-purple-500',
    'from-purple-50', 'bg-purple-50', 'focus:ring-purple-500',
    // Hover variants
    'hover:border-teal-500', 'hover:border-blue-500', 'hover:border-emerald-500', 'hover:border-purple-500',
    'hover:text-teal-600', 'hover:text-blue-600', 'hover:text-emerald-600', 'hover:text-purple-600',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#E6FFFA',
          100: '#B2F5EA',
          200: '#81E6D9',
          300: '#4FD1C5',
          400: '#38B2AC',
          500: '#319795',
          600: '#2C7A7B',
          700: '#285E61',
          800: '#234E52',
          900: '#1D4044',
        },
        dental: {
          50: '#E6FFFA',
          100: '#B2F5EA',
          200: '#81E6D9',
          300: '#4FD1C5',
          400: '#38B2AC',
          500: '#319795',
          600: '#2C7A7B',
          700: '#285E61',
          800: '#234E52',
          900: '#1D4044',
          teal: '#4FD1C5',
          mint: '#81E6D9',
          light: '#E6FFFA',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'dental-pattern': "url(\"data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.15'%3E%3Cpath d='M30 10c-3.5 0-6 2.5-6 5 0 1.5.5 2.5 1 3.5.5 1 1 2 1 3.5 0 2-1 4-1 6 0 1.5 1 2 2 2s2-.5 2-2v-4h2v4c0 1.5 1 2 2 2s2-.5 2-2c0-2-1-4-1-6 0-1.5.5-2.5 1-3.5.5-1 1-2 1-3.5 0-2.5-2.5-5-6-5z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E\")",
      },
    },
  },
  plugins: [],
};
