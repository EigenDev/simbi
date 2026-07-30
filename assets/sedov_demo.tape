# =============================================================================
# sedov_demo.tape — the README's terminal demo, scripted so it can be re-recorded
#
#   vhs assets/sedov_demo.tape        (run from the repo root)
#
# scripted rather than performed: when the dashboard changes, re-run this instead of
# re-typing the demo and hoping the pauses land in the same places.
# =============================================================================

Output assets/sedov.gif

Set Shell zsh
Set FontSize 14
Set Width 1200
Set Height 760
Set Padding 12
Set TypingSpeed 90ms

# off-camera setup: activate the venv and clear, so the recording opens on a bare prompt
Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 2s
Show

# the shot
Type "simbi run sedov"
Sleep 800ms
Enter

# let the dashboard paint and the run get going. tune this: the sedov default is
# end_time = 1.0 on a 2D spherical-polar grid sized from zones-per-decade, so if the run
# finishes early the TUI exits on its own and the `q` below lands harmlessly at the shell.
Sleep 25s

# quit the way a user would — `q` writes a restartable `interrupted` checkpoint and
# restores the terminal
Type "q"
Sleep 2s
