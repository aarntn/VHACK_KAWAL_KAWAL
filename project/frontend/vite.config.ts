import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  css: {
    // Use postcss (with tailwindcss plugin) to expand @tailwind directives
    // before any minifier sees them.  lightningcss then receives plain CSS
    // and no longer emits "Unknown at rule: @tailwind" warnings.
    transformer: 'postcss',
  },
})
