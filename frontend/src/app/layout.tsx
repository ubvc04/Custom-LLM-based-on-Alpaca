import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'
import { Toaster } from 'sonner'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '🏆 Alpaca Domination - World\'s Best LLM',
  description: 'The most advanced Alpaca-trained language model achieving #1 global performance',
  keywords: ['AI', 'LLM', 'Alpaca', 'ChatGPT', 'Language Model', 'Machine Learning'],
  authors: [{ name: 'Alpaca Domination Team' }],
  openGraph: {
    title: 'Alpaca Domination - World\'s Best LLM',
    description: 'Experience the future of AI conversation with our state-of-the-art model',
    type: 'website',
    images: ['/og-image.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Alpaca Domination - World\'s Best LLM',
    description: 'Experience the future of AI conversation',
    images: ['/og-image.png'],
  },
  viewport: 'width=device-width, initial-scale=1',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0f172a' },
  ],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          {children}
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: 'var(--toast-bg)',
                color: 'var(--toast-color)',
                border: '1px solid var(--toast-border)',
              },
            }}
          />
        </Providers>
      </body>
    </html>
  )
}
