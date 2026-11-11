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


// ===
// this section I generated with gemini
// ===
const modules = import.meta.glob('./markdowns/*.md', { query: '?raw', import: 'default' });

for (const [path, loader] of Object.entries(modules)) {

  const name = path.split('/').pop()?.replace('.md', '');
  const el = document.getElementById(`markdown-${name}`);

  if (el) {
    const rawText = (await loader()) as string;
    el.innerHTML = await marked.parse(rawText);
  }
}

const navigation = document.getElementById('chapter-navigation');
const markdownBlocks = [...document.querySelectorAll<HTMLElement>('[id^="markdown-"]')];
const content = document.querySelector<HTMLElement>('.journey-content');
const journeyPage = document.querySelector<HTMLElement>('.journey-page');
const chapterToggle = document.querySelector<HTMLButtonElement>('.chapter-toggle');

chapterToggle?.addEventListener('click', () => {
  const isOpen = journeyPage?.classList.toggle('chapters-open') ?? false;
  chapterToggle.setAttribute('aria-expanded', String(isOpen));
  chapterToggle.querySelector('[aria-hidden]')!.textContent = isOpen ? '←' : '→';
});

if (navigation && markdownBlocks.length > 0) {
  const headings = markdownBlocks.flatMap((block) =>
    [...block.querySelectorAll<HTMLHeadingElement>('h2, h3')]
  );
  const usedIds = new Set<string>();

  const createHeadingId = (heading: HTMLHeadingElement) => {
    const baseId = (heading.textContent ?? 'chapter')
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/[\s_-]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'chapter';
    let id = baseId;
    let suffix = 2;

    while (usedIds.has(id) || document.getElementById(id)) {
      id = `${baseId}-${suffix}`;
      suffix += 1;
    }

    usedIds.add(id);
    heading.id = id;
    return id;
  };

  const chapterList = document.createElement('ul');
  let subchapterList: HTMLUListElement | undefined;
  const chapterItemByHeadingId = new Map<string, HTMLLIElement>();

  for (const heading of headings) {
    const link = document.createElement('a');
    const headingId = createHeadingId(heading);
    link.href = `#${headingId}`;
    link.textContent = heading.textContent ?? 'Chapter';

    const item = document.createElement('li');
    item.append(link);

    if (heading.tagName === 'H2') {
      chapterList.append(item);
      subchapterList = document.createElement('ul');
      item.append(subchapterList);
      chapterItemByHeadingId.set(headingId, item);
    } else if (subchapterList) {
      subchapterList.append(item);
    } else {
      chapterList.append(item);
    }
  }

  navigation.replaceChildren(chapterList);

  const links = [...navigation.querySelectorAll<HTMLAnchorElement>('a')];
  const linkById = new Map(links.map((link) => [link.hash.slice(1), link]));
  const chapterHeadings = headings.filter((heading) => heading.tagName === 'H2');
  const updateCurrentChapter = () => {
    const contentTop = content?.getBoundingClientRect().top ?? 0;
    const readingLine = contentTop + (content?.clientHeight ?? window.innerHeight) * 0.25;
    const currentChapterHeading = chapterHeadings.reduce<HTMLHeadingElement | undefined>(
      (current, heading) =>
        heading.getBoundingClientRect().top <= readingLine ? heading : current,
      undefined
    ) ?? chapterHeadings[0];
    const currentChapter = currentChapterHeading
      ? chapterItemByHeadingId.get(currentChapterHeading.id)
      : undefined;

    for (const link of links) {
      link.removeAttribute('aria-current');
    }
    linkById.get(currentChapterHeading?.id ?? '')?.setAttribute('aria-current', 'location');
    for (const item of chapterItemByHeadingId.values()) {
      item.classList.toggle('chapter-current', item === currentChapter);
    }
  };

  updateCurrentChapter();
  content?.addEventListener('scroll', updateCurrentChapter, { passive: true });
  content?.addEventListener('resize', updateCurrentChapter);
  window.addEventListener('resize', updateCurrentChapter);
}
// ===