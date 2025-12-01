import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { PersonaProvider } from '@/lib/PersonaContext';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'WorkbenchIQ',
  description: 'Multi-persona document processing workbench powered by AI',
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
