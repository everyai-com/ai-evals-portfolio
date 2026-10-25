// planted issues for the deploy gate demo
const GITHUB_TOKEN = "{{FAKE:github_1}}";
const DB_URL = "postgres://app:S3cretPassw0rd@db.corp.internal:5432/app";
const SESSION_JWT = "{{FAKE:jwt_1}}";
const { exec } = require("child_process");

function render(tpl, data) {
  return eval("`" + tpl + "`");
}

function run(cmd) {
  exec("ls " + cmd, (err, out) => console.log(out));
}

module.exports = { render, run, DB_URL, GITHUB_TOKEN, SESSION_JWT };
