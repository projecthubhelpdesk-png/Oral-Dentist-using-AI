import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface DentalTheme {
  id: string;
  name: string;
  // Gradient colors
  gradientFrom: string;
  gradientVia: string;
  gradientTo: string;
  // Accent colors
  accent: string;
  accentHover: string;
  accentLight: string;
  accentDark: string;
  // Text colors
  textOnGradient: string;
  textMuted: string;
  // Card styles
  cardBorder: string;
  cardHeaderBg: string;
  // Button styles
  buttonBg: string;
  buttonHover: string;
  buttonText: string;
}

export const dentalThemes: DentalTheme[] = [
  {
    id: 'teal',
    name: 'Ocean Teal',
    gradientFrom: 'from-teal-400',
    gradientVia: 'via-teal-500',
    gradientTo: 'to-teal-600',
    accent: 'teal-500',
    accentHover: 'teal-600',
    accentLight: 'teal-50',
    accentDark: 'teal-800',
    textOnGradient: 'text-white',
    textMuted: 'text-white/80',
    cardBorder: 'border-teal-100',
    cardHeaderBg: 'from-teal-50 to-white',
    buttonBg: 'bg-teal-500',
    buttonHover: 'hover:bg-teal-600',
    buttonText: 'text-white',
  },
  {
    id: 'blue',
    name: 'Sky Blue',
    gradientFrom: 'from-blue-400',
    gradientVia: 'via-blue-500',
    gradientTo: 'to-indigo-600',
    accent: 'blue-500',
    accentHover: 'blue-600',
    accentLight: 'blue-50',
    accentDark: 'blue-800',
    textOnGradient: 'text-white',
    textMuted: 'text-white/80',
    cardBorder: 'border-blue-100',
    cardHeaderBg: 'from-blue-50 to-white',
    buttonBg: 'bg-blue-500',
    buttonHover: 'hover:bg-blue-600',
    buttonText: 'text-white',
  },
  {
    id: 'emerald',
    name: 'Fresh Mint',
    gradientFrom: 'from-emerald-400',
    gradientVia: 'via-emerald-500',
    gradientTo: 'to-green-600',
    accent: 'emerald-500',
    accentHover: 'emerald-600',
    accentLight: 'emerald-50',
    accentDark: 'emerald-800',
    textOnGradient: 'text-white',
    textMuted: 'text-white/80',
    cardBorder: 'border-emerald-100',
    cardHeaderBg: 'from-emerald-50 to-white',
    buttonBg: 'bg-emerald-500',
    buttonHover: 'hover:bg-emerald-600',
    buttonText: 'text-white',
  },
  {
    id: 'purple',
    name: 'Royal Purple',
    gradientFrom: 'from-purple-400',
    gradientVia: 'via-purple-500',
    gradientTo: 'to-indigo-600',
    accent: 'purple-500',
    accentHover: 'purple-600',
    accentLight: 'purple-50',
    accentDark: 'purple-800',
    textOnGradient: 'text-white',
    textMuted: 'text-white/80',
    cardBorder: 'border-purple-100',
    cardHeaderBg: 'from-purple-50 to-white',
    buttonBg: 'bg-purple-500',
    buttonHover: 'hover:bg-purple-600',
    buttonText: 'text-white',
  },
];

interface ThemeContextType {
  theme: DentalTheme;
  setTheme: (theme: DentalTheme) => void;
  randomizeTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<DentalTheme>(() => {
    // Random theme on initial load
    const randomIndex = Math.floor(Math.random() * dentalThemes.length);
    return dentalThemes[randomIndex];
  });

  const randomizeTheme = () => {
    const randomIndex = Math.floor(Math.random() * dentalThemes.length);
    setTheme(dentalThemes[randomIndex]);
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, randomizeTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
