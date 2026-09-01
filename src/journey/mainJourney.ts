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


async function loadMarkdown(url: string): Promise<string> {
    const response = await fetch(url);
    return response.text();
}

const introPath = 'src/journey/markdowns/intro.md';
const setupPath = 'src/journey/markdowns/setup.md';

const introContent = await loadMarkdown(introPath);
const setupContent = await loadMarkdown(setupPath);


const introMD : string = await marked.parse(introContent);
const setupMD : string = await marked.parse(setupContent);


document.getElementById('markdown-intro')!.innerHTML = introMD;

document.getElementById('markdown-setup')!.innerHTML = setupMD;
