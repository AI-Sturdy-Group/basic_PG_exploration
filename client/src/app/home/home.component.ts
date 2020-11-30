import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Configuration } from '../model/configuration';
import { ExperimentInfo } from '../model/experimentInfo';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  configuration: Configuration;
  experimentInfo: any = [];
  configurationForm: FormGroup;
  keys: any[];

  constructor(private http: HttpClient,
              private fb: FormBuilder) { }

  ngOnInit() {
    this.createForm();
  }

  runExperiment() {
    this.configuration = Object.assign({}, this.configurationForm.value);
    this.http.post<ExperimentInfo>('http://localhost:5000/start_training', this.configuration).subscribe(next => {
      this.experimentInfo = next;
    }, error => {
      console.log(error);
    });
  }

  createForm() {
    this.configurationForm = this.fb.group({
      name: ['test_run', Validators.required],
      desc: ['Test Experiment', Validators.required],
      training_steps: [100, Validators.min(1)],
      show_every: [5, Validators.min(0)],
      learning_rate: [0.001, Validators.min(0)],
      experience_size: [5, Validators.min(0)],
      minibatch_size: null,
      hidden_layer_sizes: [[]],
      hidden_activation: ['tanh', Validators.required],
      actions_size: [1, Validators.min(1)],
      save_policy_every: [5, Validators.min(0)],
      mu_activation: ['tanh', Validators.required],
      sigma_activation: ['softplus', Validators.required]
    });

    this.keys = Object.keys(this.configurationForm.controls);
  }

  parseResult(next: ExperimentInfo) {
    const values = Object.keys(next);

    values.forEach(element => {
      this.experimentInfo[element] = next[element];
    });
  }

  typeOfValue(argument: string) {
    return typeof(this.configurationForm.get(argument).value);
  }

}
