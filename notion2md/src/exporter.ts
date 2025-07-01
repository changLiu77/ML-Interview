import { NotionConverter } from 'notion-to-md';
import { Client } from '@notionhq/client';
import { NotionExporter, ChainData } from 'notion-to-md/types';
import { DefaultExporter } from 'notion-to-md/plugins/exporter';
import { BlockObjectResponse, ChildPageBlockObjectResponse } from '@notionhq/client/build/src/api-endpoints';
import { PageReferenceManifestBuilder } from 'notion-to-md/utils';
import * as path from 'path';

class ConsoleExporter implements NotionExporter {
  private defaultExporter: DefaultExporter;

  constructor(outputPath: string, private verbose = false) {
    this.defaultExporter = new DefaultExporter({
      outputType: 'file',
      outputPath: outputPath,
      });
  }
  async export(data: ChainData): Promise<void> {
    // console.log('-------- Converted Content --------');
    // console.log(data.content); // contains the final rendered output
    console.log('---------------------------------');
    if (this.verbose) {
      console.log('Page ID:', data.pageId);
      console.log('Content Length:', data.content.length);
      console.log('Block Count:', data.blockTree.blocks.length);
    }

    await this.defaultExporter.export(data);
  }
}


const notion = new Client({ auth: process.env.NOTION_TOKEN });

async function getPageTitle(pageID:string): Promise<string> {
  const response = await notion.pages.retrieve({ page_id: pageID });
  const properties = (response as any).properties;
  if ('title' in response) {
    return (response as any).title;
    }
  
  for (const key in properties) {
    const prop = properties[key];
    if (prop.type === 'title' && prop.title.length > 0) {
    return prop.title.map((t: { plain_text: string }) => t.plain_text).join('');
    }
  }
  throw new Error('Page title not found.');
}

async function getDatabaseIdFromMainPage(mainPageId: string): Promise<string | undefined> {
  const res = await notion.blocks.children.list({
    block_id: mainPageId,
    page_size: 100,
  });
  // console.log(res.results)
  for (const block of res.results) {
    if ('type' in block && block.type === 'child_database') {
      return block.id;
    }
  }

  return undefined;
}

async function getSubPagesFromDataset(databaseId: string) {
  const subPages: { id: string, url: string }[] = [];
  const res = await notion.databases.query({
  database_id: databaseId,
  page_size: 100,
  });
  for (const page of res.results) {
    const pageId = page.id.replace(/-/g, '');
    const urlProp = (page as any).properties['URL'];
    const url = urlProp.rich_text.map((t: { plain_text: string }) => t.plain_text).join('');
    subPages.push({ id: pageId, url });
  }
  return subPages;
}

  async function exportPage(pageId: string, outputPath: string, exporter: NotionExporter) {
    const n2m = new NotionConverter(notion)
    .withExporter(exporter)
    .downloadMediaTo({
      outputDir: outputPath,
      // Update the links in markdown to point to the local media path
      transformPath: (localPath: string) => `./media/${path.basename(localPath)}`,
      });
    await n2m.convert(pageId);
    console.log(
      `Successfully converted page  ${pageId}`,
    );
    }

async function main() {
  try{
    const pageId = '1540583430de806bb2ebfca8aac5693d';
    const outputDir = './output'; // Define where to save the file
    const mediaDir = path.join(outputDir, 'media');

    // get subpages
    const databaseId = await getDatabaseIdFromMainPage(pageId);
    if (!databaseId) {
      console.log('No Database with subpages');
      return;
    }
    
    console.log('Database ID:', databaseId);
    const subPages = await getSubPagesFromDataset(databaseId);
    console.log('Subpages:', subPages);
    
    const preBuiltPages: Record<string, string> = {};
    for (const page of subPages) {
      const outputFilePath = `./output/${page.url}`;
      const exporter = new ConsoleExporter(outputFilePath, true);
      await exportPage(page.id, mediaDir, exporter);
      preBuiltPages[page.id] = outputFilePath;
    }
    
    const builder = new PageReferenceManifestBuilder(notion, {
      urlPropertyNameNotion: 'URL',  // The name of your Notion property
      baseUrl: ''  // Your site's base URL
      });
      // Build manifest starting from a root page or database
    await builder.build(databaseId);
    console.log('Manifest built successfully!');
    // get the page title
    const title = await getPageTitle(pageId); 
    const pageTitle = title.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
    const outputFileDir = path.join(outputDir, `${pageTitle}.md`);
    const exporter = new ConsoleExporter(outputFileDir, true);
    const n2m = new NotionConverter(notion)
    .withExporter(exporter)
    .downloadMediaTo({
      outputDir: mediaDir,
      // Update the links in markdown to point to the local media path
      transformPath: (localPath: string) => `./media/${path.basename(localPath)}`,
      })
     .withPageReferences({
      baseUrl: '',
      UrlPropertyNameNotion: 'URL',
    }) 
    ;
    await n2m.convert(pageId);
    console.log(
      `✓ Successfully converted page and saved to ${outputFileDir}`,
    );
  } catch (error) {
    console.error('Conversion failed:', error);
  }
}

main();
