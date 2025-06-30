import { NotionConverter } from 'notion-to-md';
import { Client } from '@notionhq/client';
import { NotionExporter, ChainData } from 'notion-to-md/types';

class ConsoleExporter implements NotionExporter {
  constructor(private verbose = false) {}
  async export(data: ChainData): Promise<void> {
    console.log('-------- Converted Content --------');
    console.log(data.content); // contains the final rendered output
    console.log('---------------------------------');
    if (this.verbose) {
      console.log('Page ID:', data.pageId);
      console.log('Content Length:', data.content.length);
      console.log('Block Count:', data.blockTree.blocks.length);
    }
  }
}
const notion = new Client({ auth: process.env.NOTION_TOKEN });

const n2m = new NotionConverter(notion)
.withExporter(new ConsoleExporter(true));

async function main() {
  await n2m.convert('1540583430de806bb2ebfca8aac5693d');
}

main();
