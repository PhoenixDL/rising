## Short Description

Please give a short summary of the main points of this PR

## PR Checklist

### PR Implementer

This is a small checklist for the implementation details of this PR.
If you submit a PR, please look at these points (don't worry about the `RisingTeam`
and `Reviewer` workflows, the only purpose of those is to have a compact view of
the steps)

If there are any questions regarding code style or other conventions check out our
[summary](https://github.com/PhoenixDL/rising/blob/master/CONTRIBUTING.md).

- [ ] Implementation
- [ ] Docstrings & Typing
- [ ] Check `__all__` sections and `__init__`
- [ ] Unittests (look at the line coverage of your tests, the goal is 100%!)
- [ ] Update notebooks & documentation if necessary
- [ ] Pass all tests
- [ ] Add the checksum of the last implementation commit to the Changelog

### RisingTeam

<details>
  <summary>RisingTeam workflow</summary>

- [ ] Add pull request to project (optionally delete corresponding project note)
- [ ] Assign correct label (if you don't have permission to do this, someone will do it for you.
  Please make sure to communicate the current status of the pr.)
- [ ] Does this PR close an Issue? (add `closes #IssueNumber` at the bottom if
  not already in description)

</details>

### Reviewer

<details>
  <summary>Reviewer workflow</summary>

- [ ] Do all tests pass? (Unittests, NotebookTests, Documentation)
- [ ] Does the implementation follow `rising` design conventions?
- [ ] Are the tests useful? (!!!) Are additional tests needed?
  Can you think of critical points which should be covered in an additional test?
- [ ] Optional: Check coverage locally / Check tests locally if GPU is necessary to execute

</details>
