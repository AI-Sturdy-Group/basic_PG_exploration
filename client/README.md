# Client

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 9.0.7.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

# First Run

Execute `ng update`

# To Run

Go to client folder from terminal and execute: `ng serve`

# To Build

Run `ng build --prod` from terminal to generate the production version of the app, witch, for now, will be located in folder: `./dist/client`

# To use production version

Access `index.html` file from its root with flask server.

Take into consideration that from Angular 8 foward you will need to add the line
`if path.endswith(".js"): return send_from_directory('./client/dist/client/', path, mimetype="application/javascript")`. Otherwise it will result in a MIME error.