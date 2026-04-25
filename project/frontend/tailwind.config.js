/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'figma-bg': '#010101',
        'figma-body-bg': '#10141A',
        'figma-accent': '#92B6FF',
        'figma-card': '#1C2026',
        'figma-card-border': 'rgba(66, 71, 84, 0.05)',
        'figma-text-muted': '#C2C6D6',
        'figma-text-secondary': '#D2D2D2',
        'figma-safe': '#10B981',
        'figma-safe-light': 'rgba(16, 185, 129, 0.1)',
        'figma-warning': '#EF4444',
      },
    },
  },
  plugins: [],
}