import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Persistent Mind Model',
  description: 'Architecture, API, and UI docs',
  lastUpdated: true,
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/intro' },
      { text: 'API', link: '/api/probe' },
      { text: 'UI', link: '/ui/overview' },
    ],
    sidebar: {
      '/guide/': [
        { text: 'Introduction', link: '/guide/intro' },
        { text: 'Getting Started', link: '/guide/getting-started' },
        { text: 'Development Workflow', link: '/guide/dev-workflow' },
  { text: 'Architecture', link: '/guide/architecture' },
  { text: 'LangChain Integration', link: '/guide/langchain' },
      ],
      '/api/': [
        { text: 'Probe API', link: '/api/probe' },
      ],
      '/ui/': [
        { text: 'Overview', link: '/ui/overview' },
        { text: 'Data Flow', link: '/ui/data-flow' },
      ]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/scottonanski/persistent-mind-model' }
    ]
  }
})
