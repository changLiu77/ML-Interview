import { NotionConverter } from 'notion-to-md';
import { Client } from '@notionhq/client';
import { NotionExporter, ChainData } from 'notion-to-md/types';
import { DefaultExporter } from 'notion-to-md/plugins/exporter';
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


async function main() {
  try{
    const pageId = '1540583430de806bb2ebfca8aac5693d';
    const outputDir = './output'; // Define where to save the file
    const mediaDir = path.join(outputDir, 'media');
    // Configure the DefaultExporter to save to a file
    const exporter = new ConsoleExporter(path.join(outputDir, 'page.md'), true);
    const n2m = new NotionConverter(notion)
    .withExporter(exporter)
    .downloadMediaTo({
      outputDir: mediaDir,
      // Update the links in markdown to point to the local media path
      transformPath: (localPath) => `./media/${path.basename(localPath)}`,
      });
    await n2m.convert(pageId);
    console.log(
      `✓ Successfully converted page and saved to ${outputDir}/${pageId}.md`,
    );
  } catch (error) {
    console.error('Conversion failed:', error);
  }
}

main();
