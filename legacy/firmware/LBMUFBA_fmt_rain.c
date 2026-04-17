#include<stdio.h>
#include<string.h>
int main (void){
	int c,i=0,j=0,n=0;
	char a = ' ';
	FILE * pa, * pb ; /* declaracao do ponteiro para arquivo */
	char * nome = "../LBM_rain_2019.dat";
     /* nome externo associado ao interno */
     pa = fopen (nome , "r" );
     if ( pa == NULL ) { /* verifica erro na abertura */
     printf("Arquivo nao pode ser aberto.\n" );
     return 1;
     }
     else{
		printf("Lendo o arquivo!");
	 }
	 if (( pb = fopen ("../formatado/LBMUFBA_rain.dat", "w" )) == NULL ) { /* Abre o arquivo ../formatado/LBM_rain.dat para salvar o arquivo compilado */
		printf ("\nErro ao abrir o arquivo para escrita.\n" );
		return 1;
	}
	 while (n<237) { /* Exclui as primeiras linhas do arquivo gerado pelo datalogger */
         c = fgetc (pa);
         n++;
         }
         while (!feof (pa)) {
                if (c=='I'){ /* Substitui INF por -999.999 */
                c = fgetc (pa);
                       if (c=='N'){
                       c = fgetc (pa);
                               if (c=='F'){
                               c = fgetc (pa);
                               fprintf(pb, "-999.999");
                               }
                               else{
                               fprintf(pb, "IN");
                               }
                       }
                       else{
                       fprintf(pb, "I");
                       }
                }
				if (c=='N'){ /* Substitui NAN por -999.999 */
                c = fgetc (pa);
                       if (c=='A'){
                       c = fgetc (pa);
                               if (c=='N'){
                               c = fgetc (pa);
                               fprintf(pb, "-999.999");
                               }
                               else{
                               fprintf(pb, "NA");
                               }
                       }
                       else{
                       fprintf(pb, "N");
                       }
                }
			if ((c!=',') & (c!='"') & (c!=':')){ /*Salva o caracter no arquivo se for um número*/
			fputc (c, pb );
			c = fgetc (pa);
				if (c=='-'){ /*Salva apenas os traços utilizados como sinais negativos*/
				j++;	
					while(j<3){
						if (c=='-'){
						c=a;	
						fputc (c, pb );
						j++;
						}
						else{
						fputc (c, pb );
						}
					c = fgetc (pa);	
					}
				}
			}
			else{ /* Se não for um número conta-se o número de virgulas para salvar apenas as variáveis desejadas*/
	                if (c==','){  
					c=a;	
					fputc (c, pb );
					i++;
						while(i<0){
						c = fgetc (pa);
							if (c==','){
							i++;
							}
						}
						if (i==1){  
							while(i<2){
							c = fgetc (pa);
								if (c==','){
								i++;
								}
							}
						}
						if (i==3){ 
							while(i<6){
							c = fgetc (pa);
								if (c==','){ 
								i++;
								}
							}
						}
						if (i==6){ 
						i=0;
						j=0;
						}
					c = fgetc (pa);		
					}
					else{ /* Substitui qualquer outro caracter por espaço */
					c=a;	
					fputc (c, pb );
					c = fgetc (pa);
					}
			}
		}
	 printf ("\nArquivo dados_compilados gravado.\n" );
	 fclose (pb); /* Fecha o arquivo LBM_rain.dat */
	 fclose (pa); /* Fecha o arquivo LBM_rain.dat */
 return 0;
}
