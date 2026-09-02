import { Sandbox as CloudflareSandbox } from "@cloudflare/sandbox";
import { bridge } from "@cloudflare/sandbox/bridge";

export { WarmPool } from "@cloudflare/sandbox/bridge";

export class Sandbox extends CloudflareSandbox {
	override sleepAfter = "30m";
}

export default bridge({
	async fetch(): Promise<Response> {
		return new Response("BenchMax AIME Cloudflare Sandbox bridge");
	},
});
