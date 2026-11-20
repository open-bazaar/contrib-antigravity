# bazaar-contrib-antigravity

Antigravity Harness provider for [Bazaar Compute Node](https://github.com/yuchanns/bazaar-compute-node).

## Setup

Install and sign in to [Antigravity CLI](https://antigravity.google/download) 1.5 or later, then install BCN and this provider.

```sh
uv tool install bazaar-compute-node
bcn contrib add bazaar-contrib-antigravity --enable
```

Add an agent using an installed Channel provider.

```sh
bcn agent add --name Editor --channel CHANNEL_KIND --harness antigravity
bcn run
```

## Configuration

Optional fields in the agent's `antigravity` Harness configuration.

- `model` selects the Antigravity model and defaults to the CLI configuration.
- `effort` selects the reasoning effort and defaults to the CLI configuration.
- `api_key` supplies the Gemini API key when the CLI's `modelProvider` is set to `gemini`.
