import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import { createHighlighter } from 'shiki';
import markedShiki from 'marked-shiki';
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
  transformerNotationFocus,
  transformerNotationErrorLevel,
  transformerMetaHighlight,
  transformerMetaWordHighlight
} from '@shikijs/transformers'

// Initialize Shiki
const highlighter = await createHighlighter({
  themes: ['github-dark-dimmed'],
  langs: ['typescript', 'javascript', 'html', 'css', 'glsl', 'wgsl', `markdown`]
});

// init marked (markdown parser, copied from docs: https://www.npmjs.com/package/marked-shiki)
marked
.use(
    markedKatex({
      throwOnError: false,
      nonStandard: true,
    })
  )
.use(markedShiki({
  highlight(code, lang, props) {
    return highlighter.codeToHtml(code, {
      lang,
      theme: 'github-dark-dimmed',
      meta: { __raw: props.join(' ') }, // required by `transformerMeta*`
      transformers: [
        transformerNotationDiff({
          matchAlgorithm: 'v3'
        }),
        transformerNotationHighlight({
          matchAlgorithm: 'v3'
        }),
        transformerNotationWordHighlight({
          matchAlgorithm: 'v3'
        }),
        transformerNotationFocus({
          matchAlgorithm: 'v3'
        }),
        transformerNotationErrorLevel({
          matchAlgorithm: 'v3'
        }),
        transformerMetaHighlight(),
        transformerMetaWordHighlight()
      ]
    })
  }
}
)); 

import introRaw from './markdowns/intro.md?raw';
import setupRaw from './markdowns/setup.md?raw';
import splatRaw from './markdowns/splat.md?raw';
import splatEditRaw from './markdowns/splatEdit.md?raw';
import harmonicsRaw from './markdowns/sphericalHarmonics.md?raw';


interface MarkdownSection {
  id: string;
  rawText: string;
}

const markdownSections: MarkdownSection[] = [
  { id: 'markdown-intro', rawText: introRaw },
  { id: 'markdown-setup', rawText: setupRaw },
  { id: 'markdown-drawgauss', rawText: splatRaw },
  { id: 'markdown-elipsoid', rawText: splatEditRaw },
  { id: 'markdown-harmonics', rawText: harmonicsRaw },
];

for (const { id, rawText } of markdownSections) {
  const el = document.getElementById(id);
  if (el) {
    el.innerHTML = await marked.parse(rawText);
  }
}
