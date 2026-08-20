# Third-party robot models

## Franka Emika Panda

- Source: MuJoCo Menagerie
- Repository: https://github.com/google-deepmind/mujoco_menagerie
- Commit: `da76818e269b82289eba39808e2fb91d679d6994`
- Upstream directory: `franka_emika_panda`
- License: Apache-2.0; see `franka_emika_panda/LICENSE`

The upstream model files are copied without modification. The `pilot_pick_place.xml`
and `pilot_peg_insert.xml` files are local scene definitions that include the upstream
`mjx_panda.xml` robot model.
