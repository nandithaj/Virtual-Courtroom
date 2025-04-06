# Frontend Repository: React + Vite + TypeScript with MUI & Tailwind CSS

This repository contains the frontend for our project, built with React, Vite, and TypeScript. It comes pre-configured with Material UI (MUI) for component styling and Tailwind CSS for utility-first CSS styling. Follow the steps below to set up and run the project(Note use this app for our dedicated backend of VeinChain from the repo named-VeinChain).

---

## Prerequisites

- [Node.js](https://nodejs.org/) (v14 or higher recommended)
- [npm](https://www.npmjs.com/) (or yarn, if preferred)

---

## Getting Started

### 1. Install Dependencies

Clone the repository and install all necessary dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
npm install
```

This will install React, Vite, TypeScript, MUI, Tailwind CSS, and other required packages.

### 2. Configure Tailwind CSS

Tailwind CSS is already installed. Verify or update your Tailwind configuration in `tailwind.config.js` if needed. For a basic configuration, your file should look similar to:

```js
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

Also, ensure that Tailwind’s directives are included in your CSS. Typically, your main CSS file (e.g., `src/index.css`) should contain:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 3. Configure Material UI (MUI)

Material UI is integrated into the project. Import MUI components as needed in your React components. For example:

```tsx
import Button from '@mui/material/Button';

function App() {
  return (
    <div className="p-4">
      <Button variant="contained" color="primary">
        Hello MUI!
      </Button>
    </div>
  );
}

export default App;
```

---

## Running the Application

Start the development server with Hot Module Replacement (HMR) enabled:

```bash
npm run dev
```

Open your browser and navigate to the URL provided in the terminal (typically [http://localhost:3000](http://localhost:3000)) to see your app in action.

---

## Building for Production

To create an optimized production build:

```bash
npm run build
```

The output will be generated in the `dist` folder.

---

## Linting & Formatting

For code quality, the project includes ESLint and Prettier configurations. You can run the linter using:

```bash
npm run lint
```

If you need to fix linting issues automatically, run:

```bash
npm run lint:fix
```

---


### 4.Change the backend url as per the dynamic one 

Change the url for the api services from the backend accordingly.

## Additional Configuration: ESLint with React & TypeScript

If you wish to expand your ESLint configuration for a production-level application, consider using plugins for React and TypeScript. An example configuration in `eslint.config.js` might be:

```js
import tseslint from '@typescript-eslint/eslint-plugin';
import reactX from 'eslint-plugin-react-x';
import reactDom from 'eslint-plugin-react-dom';

export default tseslint.config({
  extends: [
    ...tseslint.configs.recommendedTypeChecked,
    ...tseslint.configs.strictTypeChecked,
    ...tseslint.configs.stylisticTypeChecked,
  ],
  languageOptions: {
    parserOptions: {
      project: ['./tsconfig.json'],
      tsconfigRootDir: __dirname,
    },
  },
  plugins: {
    'react-x': reactX,
    'react-dom': reactDom,
  },
  rules: {
    ...reactX.configs['recommended-typescript'].rules,
    ...reactDom.configs.recommended.rules,
  },
});
```

Adjust the configuration as needed for your project's coding standards.

---

## Conclusion

You now have everything you need to get started with the frontend repository. Enjoy developing with React, Vite, TypeScript, MUI, and Tailwind CSS!

For any issues or further customization, please refer to the official documentation of [React](https://reactjs.org/), [Vite](https://vitejs.dev/), [TypeScript](https://www.typescriptlang.org/), [Material UI](https://mui.com/), and [Tailwind CSS](https://tailwindcss.com/).

Happy coding!