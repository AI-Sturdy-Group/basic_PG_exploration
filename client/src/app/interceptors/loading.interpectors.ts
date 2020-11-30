import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { finalize } from 'rxjs/operators';
import { BusyService } from 'src/app/services/busy.service';

@Injectable()
export class LoadingInterceptors implements HttpInterceptor {
    constructor(private busyService: BusyService) {}

    intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
        this.busyService.busy();
        return next.handle(req).pipe(
            finalize(() => {
                this.busyService.idle();
            })
        );
    }

}
