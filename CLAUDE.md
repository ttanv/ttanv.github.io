# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Jekyll-based personal academic website hosted on GitHub Pages. It's a fork of Jekyll Now, styled after Jon Barron's academic website.

## Build Commands

```bash
# Local development (requires Jekyll installed)
bundle exec jekyll serve

# Generate thumbnails for new images (requires ImageMagick)
./_make_thumbnails.sh
```

## Architecture

### Content Structure

- **`_posts/`**: Markdown files for content entries. Each post uses front matter to define metadata:
  - `categories`: Determines display section (`research`, `Intel` for personal projects, or other for coursework)
  - `image`: Path to project image (displayed as thumbnail from `tn/` directory)
  - `authors`, `venue`, `arxiv`, `code`, `video`, etc.: Optional metadata for links

- **`_layouts/default.html`**: Single-page layout that renders all content. Posts are filtered by category into three sections:
  1. Research (category: `research`)
  2. Personal Projects (category: `Intel`)
  3. Other Projects (all other categories)

### Image Handling

Images go in `images/`. Thumbnails are stored in `tn/images/` and generated at 160x160 using `_make_thumbnails.sh`. Templates reference thumbnails via `/tn{{post.image}}`.

### Key Config (`_config.yml`)

- `permalink: /` forces all posts to render on the homepage (no individual post pages)
- Uses Kramdown with GitHub Flavored Markdown
- Jekyll-sitemap plugin enabled
