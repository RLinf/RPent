# RoboCasa navview patch

`robosuite_navview.patch` targets
`robosuite/models/assets/bases/omron_mobile_base.xml` from the currently
audited upstream snapshot `RLinf/robosuite@85abee228d1c43ab1939bce33028099945d453b4`.
It adds the base-mounted `navview` camera required by the RoboCasa runtime.

From the root of a clean robosuite checkout:

```bash
git apply --check /path/to/robosuite_navview.patch
git apply /path/to/robosuite_navview.patch
```

Do not apply the patch twice. The reproduction `doctor` command verifies that
exactly one `navview` camera exists and that all four camera attributes match.
The patch identifies the audited input snapshot; a formal release must also
publish and freeze its complete runtime and robosuite revisions.
