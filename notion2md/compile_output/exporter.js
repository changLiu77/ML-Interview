"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const notion_to_md_1 = require("notion-to-md");
const client_1 = require("@notionhq/client");
const exporter_1 = require("notion-to-md/plugins/exporter");
const utils_1 = require("notion-to-md/utils");
const path = __importStar(require("path"));
class ConsoleExporter {
    constructor(outputPath, verbose = false) {
        this.verbose = verbose;
        this.defaultExporter = new exporter_1.DefaultExporter({
            outputType: 'file',
            outputPath: outputPath,
        });
    }
    async export(data) {
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
const notion = new client_1.Client({ auth: process.env.NOTION_TOKEN });
async function getPageTitle(pageID) {
    const response = await notion.pages.retrieve({ page_id: pageID });
    const properties = response.properties;
    if ('title' in response) {
        return response.title;
    }
    for (const key in properties) {
        const prop = properties[key];
        if (prop.type === 'title' && prop.title.length > 0) {
            return prop.title.map((t) => t.plain_text).join('');
        }
    }
    throw new Error('Page title not found.');
}
async function getDatabaseIdFromMainPage(mainPageId) {
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
async function getSubPagesFromDataset(databaseId) {
    const subPages = [];
    const res = await notion.databases.query({
        database_id: databaseId,
        page_size: 100,
    });
    for (const page of res.results) {
        const pageId = page.id.replace(/-/g, '');
        const urlProp = page.properties['URL'];
        const url = urlProp.rich_text.map((t) => t.plain_text).join('');
        subPages.push({ id: pageId, url });
    }
    return subPages;
}
async function exportPage(pageId, outputPath, exporter) {
    const n2m = new notion_to_md_1.NotionConverter(notion)
        .withExporter(exporter)
        .downloadMediaTo({
        outputDir: outputPath,
        // Update the links in markdown to point to the local media path
        transformPath: (localPath) => `./media/${path.basename(localPath)}`,
    });
    await n2m.convert(pageId);
    console.log(`Successfully converted page  ${pageId}`);
}
async function main() {
    try {
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
        const preBuiltPages = {};
        for (const page of subPages) {
            const outputFilePath = `./output/${page.url}`;
            const exporter = new ConsoleExporter(outputFilePath, true);
            await exportPage(page.id, mediaDir, exporter);
            preBuiltPages[page.id] = outputFilePath;
        }
        const builder = new utils_1.PageReferenceManifestBuilder(notion, {
            urlPropertyNameNotion: 'URL', // The name of your Notion property
            baseUrl: '' // Your site's base URL
        });
        // Build manifest starting from a root page or database
        await builder.build(databaseId);
        console.log('Manifest built successfully!');
        // get the page title
        const title = await getPageTitle(pageId);
        const pageTitle = title.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
        const outputFileDir = path.join(outputDir, `${pageTitle}.md`);
        const exporter = new ConsoleExporter(outputFileDir, true);
        const n2m = new notion_to_md_1.NotionConverter(notion)
            .withExporter(exporter)
            .downloadMediaTo({
            outputDir: mediaDir,
            // Update the links in markdown to point to the local media path
            transformPath: (localPath) => `./media/${path.basename(localPath)}`,
        })
            .withPageReferences({
            baseUrl: '',
            UrlPropertyNameNotion: 'URL',
        });
        await n2m.convert(pageId);
        console.log(`✓ Successfully converted page and saved to ${outputFileDir}`);
    }
    catch (error) {
        console.error('Conversion failed:', error);
    }
}
main();
