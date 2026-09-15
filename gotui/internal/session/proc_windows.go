//go:build windows

package session

import (
	"os/exec"
	"syscall"
)

// processGroupAttr puts the server in its own process group so Ctrl+C in the TUI
// does not also interrupt it before Stop() can run.
func processGroupAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// terminateProcessGroup kills the server process.
//
// Windows has no process-group SIGTERM equivalent that uvicorn handles, so this
// is a hard kill. Stop() already allows five seconds for a graceful exit before
// calling it, and the alternative -- leaving the port bound -- is worse.
func terminateProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
}
