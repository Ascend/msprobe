const DEVELOPMENT_PLUGIN_BASE_URL = '/data/plugin/graph_ascend';

export function buildRequestUrl(url: string, isDevelopment: boolean): string {
  const normalizedUrl = url.replace(/^\/+/, '');
  if (isDevelopment) {
    return `${DEVELOPMENT_PLUGIN_BASE_URL}/${normalizedUrl}`;
  }
  return normalizedUrl;
}
