#!/usr/bin/env bash
#
# project-git.sh — a friendly git dashboard for any project.
#
# TEMPLATE. Copy this file into the root of a repository, edit the four
# settings in the CONFIGURE ME block below, then:
#
#     chmod +x project-git.sh      # first time only
#     ./project-git.sh
#
# Menu-driven wrapper around the everyday git commands, so the normal flow
# needs no typing of raw git. Nothing here force-pushes, deletes history, or
# touches the main branch directly. Every action that changes something asks
# you to confirm first.

set -u

# ============================================================================
#  CONFIGURE ME — edit these four lines for each project, nothing else.
# ============================================================================

# Name of your project, shown at the top of the menu.
# e.g. "Atlas", "Climate Model", "My Thesis"
# Leave as-is to use the name of the project folder.
PROJECT_NAME="{{PROJECT_NAME}}"
# The GitHub repository, as owner/name. e.g. "myaccount/myrepo"
# Leave as-is to have the script read it from the 'origin' remote.
REPO="{{OWNER/REPO}}"

# The stable branch this project uses: usually "main", sometimes "master".
# Leave as-is to have the script detect it automatically.
MAIN="{{MAIN_BRANCH}}"

# Path to the working folder, shown in the "Run it" hint only.
# e.g. "~/Desktop/WORKSPACE/myproject"
WORKDIR_HINT="{{/Users/antonionavarra/Dropbox (CMCC)/ZapataLibrary/Zapata}}"

# ============================================================================
#  Nothing below here needs editing.
# ============================================================================

# ---- colours --------------------------------------------------------------
if [ -t 1 ]; then
  B="\033[1m"; DIM="\033[2m"; G="\033[32m"; Y="\033[33m"; R="\033[31m"; C="\033[36m"; Z="\033[0m"
else
  B=""; DIM=""; G=""; Y=""; R=""; C=""; Z=""
fi

say()  { printf "%b\n" "$*"; }
ok()   { printf "%b\n" "${G}✔ $*${Z}"; }
warn() { printf "%b\n" "${Y}! $*${Z}"; }
err()  { printf "%b\n" "${R}✗ $*${Z}"; }

pause() { printf "\n${DIM}Press Return to go back to the menu…${Z}"; read -r _; }

# Ask yes/no, default No. Returns 0 for yes.
confirm() {
  printf "%b" "${Y}$1 [y/N] ${Z}"
  read -r ans
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

# Run a git command, echoing it first so the user learns what happened.
run() {
  printf "${DIM}\$ %s${Z}\n" "$*"
  "$@"
}

# ---- sanity: are we inside the repo? --------------------------------------
cd "$(dirname "$0")" || exit 1
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  err "This folder is not a git repository."
  say "Put ${0##*/} in the root of your project and run it from there."
  [ "$WORKDIR_HINT" = "{{~/path/to/project}}" ] || say "  e.g.  cd ${WORKDIR_HINT} && ./${0##*/}"
  exit 1
fi

current_branch() { git rev-parse --abbrev-ref HEAD 2>/dev/null; }

# ---- fill in anything left unconfigured ------------------------------------

# PROJECT_NAME: fall back to the folder name.
case "$PROJECT_NAME" in
  ""|"{{PROJECT_NAME}}") PROJECT_NAME="$(basename "$(git rev-parse --show-toplevel)")" ;;
esac

# REPO: derive owner/name from the origin remote URL (https or ssh).
case "$REPO" in
  ""|"{{OWNER/REPO}}")
    _url="$(git config --get remote.origin.url 2>/dev/null || true)"
    if [ -n "$_url" ]; then
      REPO="$(echo "$_url" | sed -e 's#^git@[^:]*:##' -e 's#^https\{0,1\}://[^/]*/##' -e 's#\.git$##')"
    else
      REPO="(no 'origin' remote set)"
    fi
    ;;
esac

# MAIN: detect the default branch, preferring origin/HEAD, then main, then master.
case "$MAIN" in
  ""|"{{MAIN_BRANCH}}")
    MAIN="$(git symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')"
    if [ -z "$MAIN" ]; then
      if   git show-ref --verify --quiet refs/heads/main;   then MAIN="main"
      elif git show-ref --verify --quiet refs/heads/master; then MAIN="master"
      else MAIN="main"
      fi
    fi
    ;;
esac

# ---- header / status overview ---------------------------------------------
header() {
  clear 2>/dev/null || true
  local br dirty title pad
  br="$(current_branch)"
  dirty="$(git status --porcelain | wc -l | tr -d ' ')"
  title="${PROJECT_NAME} - git dashboard"
  # centre the title inside a 46-character box (trim if the name is very long)
  [ "${#title}" -gt 44 ] && title="${title:0:44}"
  pad=$(( (46 - ${#title}) / 2 )); [ "$pad" -lt 1 ] && pad=1
  say "${C}${B}╭──────────────────────────────────────────────╮${Z}"
  printf "${C}${B}│%*s%-*s│${Z}\n" "$pad" "" "$((46 - pad))" "$title"
  say "${C}${B}╰──────────────────────────────────────────────╯${Z}"
  say "  repo    : ${REPO}"
  if [ "$br" = "$MAIN" ]; then
    say "  branch  : ${Y}${br}${Z}  ${DIM}(stable — don't commit here directly)${Z}"
  else
    say "  branch  : ${G}${br}${Z}"
  fi
  if [ "$dirty" = "0" ]; then
    say "  changes : ${DIM}none — working tree clean${Z}"
  else
    say "  changes : ${Y}${dirty} file(s) changed, not yet committed${Z}"
  fi
  # ahead/behind vs its upstream, if any
  if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    local ab behind ahead
    ab="$(git rev-list --left-right --count @{u}...HEAD 2>/dev/null)"
    behind="$(echo "$ab" | awk '{print $1}')"; ahead="$(echo "$ab" | awk '{print $2}')"
    say "  vs GitHub: ${ahead} to push, ${behind} to pull"
  else
    say "  vs GitHub: ${DIM}branch not published yet${Z}"
  fi
  say ""
}

# ---- 1. open a new branch --------------------------------------------------
new_branch() {
  header
  say "${B}Open a new branch${Z}"
  say "${DIM}Starts fresh from an up-to-date ${MAIN}.${Z}\n"
  printf "Name for the new branch (e.g. feat-search): "
  read -r name
  [ -z "$name" ] && { warn "No name given — cancelled."; pause; return; }
  # sanitise: spaces -> dashes
  name="$(echo "$name" | tr ' ' '-' )"

  if [ -n "$(git status --porcelain)" ]; then
    warn "You have uncommitted changes on '$(current_branch)'."
    say "  Switching branches now could carry them along or block the switch."
    confirm "Continue anyway?" || { say "Cancelled."; pause; return; }
  fi

  say ""
  run git checkout "$MAIN"            || { err "Could not switch to $MAIN."; pause; return; }
  run git pull                        || warn "Pull failed (offline?). Continuing with local $MAIN."
  run git checkout -b "$name"         || { err "Could not create branch '$name' (does it exist already?)."; pause; return; }
  if confirm "Publish '$name' to GitHub now?"; then
    run git push -u origin "$name" && ok "Branch '$name' created and published." \
      || warn "Branch created locally, but publishing failed."
  else
    ok "Branch '$name' created locally. Publish later from the commit action."
  fi
  pause
}

# ---- 2. commit at end of day ----------------------------------------------
end_of_day() {
  header
  say "${B}Commit today's changes${Z}"
  local br; br="$(current_branch)"
  if [ "$br" = "$MAIN" ]; then
    err "You are on ${MAIN}. Never commit straight to ${MAIN}."
    say "Use option 1 to open a branch first, then come back."
    pause; return
  fi
  if [ -z "$(git status --porcelain)" ]; then
    ok "Nothing to commit — working tree is clean."
    pause; return
  fi

  say "\n${B}What changed:${Z}"
  run git status --short
  say ""
  if confirm "See the detailed line-by-line diff?"; then
    git --no-pager diff
    say ""
  fi

  confirm "Stage ALL these changes and commit?" || { say "Cancelled — nothing committed."; pause; return; }
  printf "Commit message (short, e.g. 'fix totals on summary page'): "
  read -r msg
  [ -z "$msg" ] && msg="work in progress $(date +%Y-%m-%d)"

  say ""
  run git add -A                       || { err "Staging failed."; pause; return; }
  run git commit -m "$msg"             || { err "Commit failed."; pause; return; }
  ok "Saved a restore point locally."

  if confirm "Push to GitHub now?"; then
    if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
      run git push && ok "Pushed to GitHub." || warn "Push failed (offline?). Your commit is safe locally."
    else
      run git push -u origin "$br" && ok "Published '$br' and pushed." || warn "Push failed."
    fi
  else
    say "${DIM}Commit stays local until you push.${Z}"
  fi
  pause
}

# ---- 3. sync branch with the main branch -----------------------------------
sync_with_main() {
  header
  say "${B}Sync your branch with ${MAIN}${Z}"
  local br; br="$(current_branch)"
  if [ "$br" = "$MAIN" ]; then
    warn "You are already on ${MAIN}. Just pulling the latest."
    run git pull; pause; return
  fi
  say "${DIM}Brings everyone's merged work from ${MAIN} into '${br}'.${Z}\n"

  if [ -n "$(git status --porcelain)" ]; then
    warn "You have uncommitted changes. Commit them first (option 2) for a clean sync."
    confirm "Continue without committing?" || { say "Cancelled."; pause; return; }
  fi

  run git checkout "$MAIN"   || { err "Could not switch to $MAIN."; pause; return; }
  run git pull               || warn "Pull failed (offline?)."
  run git checkout "$br"     || { err "Could not switch back to $br."; pause; return; }
  say ""
  if run git merge "$MAIN"; then
    ok "'$br' is now up to date with ${MAIN}."
  else
    warn "Merge hit conflicts. Git marked the clashing files."
    say "  Open them, look for <<<<<<< markers, fix, then:"
    say "    ${DIM}option 2 to commit the resolved merge,${Z}"
    say "  or if you'd rather back out entirely:"
    say "    ${DIM}\$ git merge --abort${Z}"
    if confirm "Abort the merge and return to how it was?"; then
      run git merge --abort && ok "Merge aborted — nothing changed."
    fi
  fi
  pause
}

# ---- 4. recover if something goes wrong -----------------------------------
recover() {
  while true; do
    header
    say "${B}Recover / undo${Z}\n"
    say "  ${C}a${Z}) Discard my edits to ONE file (not yet committed)"
    say "  ${C}b${Z}) Unstage a file (keep the edits, just un-add it)"
    say "  ${C}c${Z}) Discard ALL uncommitted edits (dangerous)"
    say "  ${C}d${Z}) Show recent history (last 10 commits)"
    say "  ${C}e${Z}) Undo a pushed commit (safe: makes a new 'undo' commit)"
    say "  ${C}f${Z}) Restore a deleted/changed file to its last committed state"
    say "  ${C}q${Z}) Back to main menu"
    printf "\nChoose: "
    read -r c
    case "$c" in
      a)
        run git status --short
        printf "File to discard edits for: "; read -r f
        [ -z "$f" ] && { warn "Cancelled."; pause; continue; }
        confirm "Really throw away your changes to '$f'?" && run git checkout -- "$f" && ok "Reverted '$f'."
        pause ;;
      b)
        run git status --short
        printf "File to unstage: "; read -r f
        [ -z "$f" ] && { warn "Cancelled."; pause; continue; }
        run git restore --staged "$f" && ok "Unstaged '$f' (edits kept)."
        pause ;;
      c)
        warn "This throws away EVERY uncommitted change in the working tree."
        if confirm "Are you absolutely sure?"; then
          run git reset --hard && ok "Working tree reset to last commit."
        else say "Cancelled."; fi
        pause ;;
      d)
        git --no-pager log --oneline -10
        pause ;;
      e)
        git --no-pager log --oneline -10
        printf "\nCommit id to undo (from the list above): "; read -r cm
        [ -z "$cm" ] && { warn "Cancelled."; pause; continue; }
        confirm "Create an undo-commit that reverses '$cm'?" && run git revert "$cm"
        pause ;;
      f)
        printf "File to restore to last committed version: "; read -r f
        [ -z "$f" ] && { warn "Cancelled."; pause; continue; }
        confirm "Restore '$f' from the last commit?" && run git restore --source=HEAD -- "$f" && ok "Restored '$f'."
        pause ;;
      q|Q) return ;;
      *) warn "Unknown choice."; sleep 1 ;;
    esac
  done
}

# ---- extras ----------------------------------------------------------------
show_status() {
  header
  say "${B}Full status${Z}\n"
  run git status
  say ""
  say "${B}Recent commits:${Z}"
  git --no-pager log --oneline -8
  pause
}

# ---- main menu -------------------------------------------------------------
while true; do
  header
  say "  ${C}1${Z}) Open a new branch"
  say "  ${C}2${Z}) Commit today's changes  ${DIM}(end of day)${Z}"
  say "  ${C}3${Z}) Sync my branch with ${MAIN}"
  say "  ${C}4${Z}) Recover / undo something"
  say "  ${C}5${Z}) Show full status"
  say "  ${C}q${Z}) Quit"
  printf "\nChoose: "
  read -r choice
  case "$choice" in
    1) new_branch ;;
    2) end_of_day ;;
    3) sync_with_main ;;
    4) recover ;;
    5) show_status ;;
    q|Q) say "Bye."; exit 0 ;;
    *) warn "Type 1, 2, 3, 4, 5 or q."; sleep 1 ;;
  esac
done
