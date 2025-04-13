import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Get Started',
      items: ['introduction', 'quickstart'],
    },
    {
      type: 'category',
      label: 'component',
      items: [
        'component/generator',
        'component/executor',
        'component/html-uploader',
        'component/paper-uploader',
      ],
    },
    {
      type: 'category',
      label: 'Development',
      items: [
        'development/local-setup',
        'development/roadmap',
      ],
    },
  ],
};

export default sidebars;
