import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['21d3-2402-3a80-1cb9-e3d7-f8a3-19c-2c08-6083.ngrok-free.app','3103-2409-40f3-9-139c-c5b9-e7d2-f288-b52e.ngrok-free.app'], // Add your ngrok URL here
  },
})
