// The same app, shaped for the platform: no server, one exported handler, static files in public/.
const OPTIONS = ['ramen', 'tacos', 'salad'];

export async function handler(request) {
  const url = new URL(request.url);
  if (url.pathname === '/api/options') {
    const r = await fetch('https://api.openai.com/v1/models', { headers: { authorization: `Bearer ${process.env.OPENAI_API_KEY}` } });
    return Response.json({ options: OPTIONS, model_count: (await r.json()).data?.length ?? 0 });
  }
  return new Response('not found', { status: 404 });
}
