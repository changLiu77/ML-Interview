import { NotionConverter } from 'notion-to-md';
import { Client } from '@notionhq/client';
import { NotionExporter, ChainData } from 'notion-to-md/types';
import { DefaultExporter } from 'notion-to-md/plugins/exporter';
import { BlockObjectResponse, ChildPageBlockObjectResponse } from '@notionhq/client/build/src/api-endpoints';
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


async function main() {
  try{
    const pageId = '1540583430de806bb2ebfca8aac5693d';
    const outputDir = './output'; // Define where to save the file
    const mediaDir = path.join(outputDir, 'media');

    // get subpages
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
      transformPath: (localPath) => `./media/${path.basename(localPath)}`,
      });
    await n2m.convert(pageId);
    console.log(
      `✓ Successfully converted page and saved to ${outputFileDir}`,
    );
  } catch (error) {
    console.error('Conversion failed:', error);
  }
}

main();
