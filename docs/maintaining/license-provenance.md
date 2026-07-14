# Maintaining Albucore License and CLA Records

Albucore's public license and its inbound Contributor License Agreement serve
different purposes:

- [`LICENSE`](../../LICENSE) is the MIT license for the public repository and
  distributed package.
- [`CLA.md`](../../CLA.md) records additional copyright, patent, authority, and
  sublicensing grants from contributors to Albumentations, LLC.

The CLA does not replace or narrow MIT permissions. Every accepted contribution
covered by the CLA must first or simultaneously appear in a public Albucore
revision under MIT before Albumentations uses the additional outbound rights.

## Versioning the CLA

Treat the root `CLA.md`, its archived copy, and the hosted CLA Assistant Gist as
one versioned offer.

1. Never edit an operative CLA version in place.
2. Create a new version and dated archive file for any substantive change.
3. Record the exact byte length and SHA-256 digest in
   [`legal/cla/archive/MANIFEST.md`](../../legal/cla/archive/MANIFEST.md).
4. Create a new immutable Gist revision containing the byte-identical CLA and
   its acceptance-field metadata.
5. Preserve prior Gist revisions and private signer exports.
6. Require contributors to accept the new version explicitly; an earlier or
   project-specific CLA acceptance does not silently carry forward.

AlbumentationsX uses a different CLA and Gist. Do not link its AGPL-specific
agreement to Albucore.

## Acceptance records

CLA Assistant handles the individual path. The hosted form must collect the
signer's full legal name and the exact versioned acceptance statement.
Authorized entity acceptances use the fields in `CLA.md` and are handled
separately through `vladimir@albumentations.ai`.

Signer exports and entity records contain personal or contractual information.
Store them in access-controlled private storage. Do not commit them to this
repository or include them in release artifacts.

## Repository and release checks

Run the source verifier before committing legal or packaging changes:

```bash
python tools/verify_legal_integrity.py
```

Build and inspect both distribution formats before a release:

```bash
uv build
python tools/verify_legal_integrity.py --artifacts dist/*.whl dist/*.tar.gz
```

The verifier enforces the MIT metadata and license bytes, the current CLA hash
and archive, packaging exclusions for inbound/private material, and the absence
of nested distribution artifacts. The `main` ruleset must require both hosted
`license/cla` and the GitHub Actions check named
`License, CLA, and package notices`.
