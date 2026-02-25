import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { PersonaProvider } from '@/lib/PersonaContext';
import { FeatureFlagsProvider } from '@/lib/FeatureFlagsContext';

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
        <FeatureFlagsProvider>
          <PersonaProvider>
            {children}
          </PersonaProvider>
        </FeatureFlagsProvider>
      </body>
    </html>
  );
}
