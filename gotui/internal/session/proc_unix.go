//go:build !windows

package session

import (
	"os/exec"
	"syscall"
)

// processGroupAttr puts the server in its own process group.
//
// This is what makes Stop() able to take the whole tree down: `python -m pythia
// serve` runs uvicorn, which can spawn its own children, and signalling only the
// parent would leave the port held.
func processGroupAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true}
}

// terminateProcessGroup signals the entire group, falling back to the single
// process if the group is already gone.
func terminateProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	// A negative pid means "every process in this group".
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM); err != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
}
