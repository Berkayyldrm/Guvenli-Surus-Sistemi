const int buzzer = 13; // the pin that the LED is attached to
int incomingByte;      // a variable to read incoming serial data into
#include <LiquidCrystal.h>
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  // initialize the LED pin as an output:
  pinMode(buzzer, OUTPUT);
  lcd.begin(16, 2);
}

void loop() {
  
  // see if there's incoming serial data:
  if (Serial.available() > 0) {
    // read the oldest byte in the serial buffer:
    incomingByte = Serial.read();
    // if it's a capital H (ASCII 72), turn on the LED:
    if (incomingByte == 'H') {
      digitalWrite(buzzer, HIGH);

      lcd.print("!!YORGUNSUNUZ!!");  
      lcd.setCursor(0, 1);
      lcd.print("Dinlenme  Vakti");
      lcd.setCursor(0,0);
      
    }
    // if it's an L (ASCII 76) turn off the LED:
    if (incomingByte == 'L') {
      digitalWrite(buzzer, LOW);

      lcd.print("!Guvenli Surus!");
      lcd.setCursor(0,1);
      lcd.print("Iyi Yolculuklar");
      lcd.setCursor(0,0);
    }
    if (incomingByte == 'S') {
      digitalWrite(buzzer, HIGH);

      lcd.print("!Sinir Uyarisi!");
      lcd.setCursor(0,1);
      
      
    }
    
  }
}
