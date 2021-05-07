#include "ESP8266WiFi.h"
#include "ESP8266WebServer.h"
#define D1 5
#define D2 4
#define D3 0
#define D4 2
#define D5 14
#define D6 12
ESP8266WebServer server(80);
 
void setup() {
  
  pinMode(D1,OUTPUT);
  pinMode(D2,OUTPUT);
  pinMode(D3,OUTPUT);
  pinMode(D4,OUTPUT);
  pinMode(D5,OUTPUT);
  pinMode(D6,OUTPUT);
  
  Serial.begin(115200);
  pinMode(2,OUTPUT);
  WiFi.begin("Devil", "123456789");  //Connect to the WiFi network
 
  while (WiFi.status() != WL_CONNECTED) {  //Wait for connection
 
    delay(500);
    Serial.println("Waiting to connect…");
 
  }
 
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());  //Print the local IP
 
  server.on("/F", forward);
  server.on("/l",left);
  server.on("/r", right);
  server.on("/s",sstop);
  
  
  server.begin();                    //Start the server
  Serial.println("Server listening");
 
}
 
void loop() {
 
  server.handleClient();         
 
}
 
void forward() {           
  
 analogWrite(D1,400);
  analogWrite(D2,400);
  digitalWrite(D3,LOW);
  digitalWrite(D4,HIGH);
  digitalWrite(D5,HIGH);
  digitalWrite(D6,LOW);
  server.send(200, "text/plain", "forward");
 
}

void left() {            
   analogWrite(D1 , 0);
  analogWrite(D2 ,300);
  digitalWrite(D3 , HIGH);
  digitalWrite(D4 , LOW);
  digitalWrite(D5, LOW);
  digitalWrite(D6, HIGH);
  
  server.send(200, "text/plain", "left");
 
}


void right() {            
   analogWrite(D1,300);
  analogWrite(D2,0);
  digitalWrite(D3,HIGH);
  digitalWrite(D4,LOW);
  digitalWrite(D5,LOW);
  digitalWrite(D6,HIGH);
  server.send(200, "text/plain", "right");
 
}

void sstop() {            
  analogWrite(D1,400);
  analogWrite(D2,400);
  digitalWrite(D3,LOW);
  digitalWrite(D4,LOW);
  digitalWrite(D5,LOW);
  digitalWrite(D6,LOW);
  server.send(200, "text/plain", "stop");
 
}