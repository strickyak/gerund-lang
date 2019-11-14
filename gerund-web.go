// A web server that executes "python ./gerund.py arg" with one arg.
// gerund.py and any .ing files that you have must be in the current directory.
package main

import (
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os/exec"
	"strings"
)

var listenFlag = flag.String("listen", ":7518", "listen host:port")

func main() {
	flag.Parse()
	pageTemplate.Parse(PAGE)
	http.HandleFunc("/", Service)
	http.ListenAndServe(*listenFlag, nil)
}

func Service(w http.ResponseWriter, r *http.Request) {
	defer func() {
		r := recover()
		if r != nil {
			log.Printf("RECOVER: %v", r)
		}
	}()
	log.Printf("Method %q URL %v", r.Method, r.URL)
	w.Header().Set("Content-Type", "text/html")

	r.ParseForm()
	text := strings.TrimSpace(r.Form.Get("text"))

	cmd := exec.Command("python2", "./gerund.py", text)
	output, err := cmd.CombinedOutput()
	result := string(output)
	if err != nil {
		result = fmt.Sprintf("ERROR: %v: %s", err, result)
	}

	d := map[string]string{
		"Text": "",
		"Pre":  result,
	}
	pageTemplate.Execute(w, d)
}

var pageTemplate = template.New("pageTemplate")

const PAGE = `
  <html><head><title>Gerund Demo</title></head><body>
    <h2>Gerund Demo</h2>

    <form method="POST" action="/">
    <textarea name=text wrap=virtual rows=5 xxxcols=40 style="width: 95%;">{{.Text}}</textarea>
    <br><input type=submit>
    </form>

    <table cellpadding=20 border=1><tr><td>
      <tt><div style="white-space: pre-wrap;">{{.Pre}}</div></tt>
    </table>

  </body>
`
