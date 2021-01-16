/*
  The game class stores all state information related to the tic tac toe game
*/
class Game {
  states: number[]
  winner: number
  agent_play: boolean
  turn: boolean
  score: number[]
  constructor() {
    // game state is an 9 index array
    // 0 <- index not taken
    // 1 <- index taken by player 1
    // 2 <- index taken by player 2 (Agent) 
    this.states = [
      0, 0, 0, 
      0, 0, 0, 
      0, 0, 0
    ]
    // if a winner is found -> (1 = player 1) (0 = player 2)
    this.winner = null
    // if player vs player or player 2 is an Agent 
    this.agent_play = false
    // who's turn it is (true(1) = player 1) (false(0) = player 2)
    this.turn = true 
    // the overall score (player 1, player 2, ties)
    this.score = [0, 0, 0]
  }

  /* update the game  */
  setState(state){
    this.states = this.states
  }

  /* Returns the current game state */
  getStates(){
    return this.states
  }

  /* Returns the winner of the game */
  getWinner(){
    return this.winner
  }

  /* Returns the current turn */
  getTurn(){
    return this.turn
  }
  
  /* Get the score of the current game */
  getScore(){
    return this.score
  }

  /* Update the state with a new move 
    - return 'win', 'tie', 'valid', 'invalid'
  */
  updateState(index){
    // check that this is a valid move (not yet taken)
    if(this.states[index] == 0) {
      // update to the new state
      if (!this.turn) {
        this.states[index] = 1
      } else {
        this.states[index] = 2
      }
      // check for winner
      if(this.checkWin()){
        // set winner status
        this.winner = Number(this.turn)
        this.updateScore()
        return 'win'
      }
      // check for tie
      if(this.checkTie()){
        this.score[2] += 1
        return 'tie'
      }
      // else change turn
      this.changeTurn()
      return 'valid'
    }
    return 'invalid'
  }

  /* Add one to the current player */
  updateScore(){
    this.score[Number(this.turn)] += 1
  }

  /* Set the game type to player vs Agent */
  playAgent(){
    this.agent_play = true
  }

  /* set the game to use the agent */
  agentPlaying(){
    return this.agent_play
  }

  /* Change the game turn to the next player */
  changeTurn(){
    // change turn
    this.turn = !this.turn
    // check if computer is playing
      // check if its the computers turn
        // get the action from the agent
  }

  /* Check for a win condition 
    - return true or false
  */
  checkWin() {
    let row = 0
    let col = 0
    for(let index = 0; index < 3; index++) {
      // if same state horizontal accross grid
      if(
          (this.states[row + 0] == this.states[row + 1]) &&
          (this.states[row + 1] == this.states[row + 2]) && 
          (this.states[row] != 0)
        ) {
        return true
      }
      // if same state vertical accross grid
      if(
          (this.states[col + 0] == this.states[col + 3]) &&
          (this.states[col + 3] == this.states[col + 6]) && 
          (this.states[col] != 0)
        ) {
        return true
      }
      col+=1
      row+=3
    }
    // if diagonal (to the right)
    if(
          (this.states[0] == this.states[4]) &&
          (this.states[4] == this.states[8]) && 
          (this.states[0] != 0)
        ) {
        return true
      }
    // if diagonal (to the left)
    if(
          (this.states[2] == this.states[4]) &&
          (this.states[4] == this.states[6]) && 
          (this.states[2] != 0)
        ) {
        return true
      }
    return false
  }

  /* Check if the game is in a tie state */
  checkTie(){
    for(let index = 0; index < this.states.length; index++) {
      if(this.states[index] == 0)
        return false
    }
    return true
  }

  /* Reset the game state */
  reset(){
    // Clear the state
    this.states = [
        0, 0, 0, 
        0, 0, 0, 
        0, 0, 0
    ]
    // set turn to player 1
    this.turn = true
    // set winner found to false
    this.winner = null
  }
} 

export const game = new Game()
