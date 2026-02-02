import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { PersonaProvider } from '@/lib/PersonaContext';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'InsureAI - Instant Life Insurance Quotes',
  description: 'AI-powered life insurance underwriting with real-time health data integration',
  icons: {
    icon: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <PersonaProvider>
          {children}
        </PersonaProvider>
      </body>
    </html>
  );
}
