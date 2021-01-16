<script>
	// Components
	import Board from "../components/Board.svelte"
	import Score from "../components/Score.svelte"
	import Turn from "../components/Turn.svelte"
	import Notification from "../components/Notification.svelte"
	// js class
	import { game } from "../game.js"

	// game states
	let states = game.getStates()
	let turn = game.getTurn()
	let score = game.getScore()
	let agent_play = false

	// notification states
	let notification 
	let show_notificaiton = false

	/* Event triggered by selecting a move */
  function selectMarker(index) {
		let outcome = game.updateState(index)
		turn = game.getTurn()
		score = game.getScore()
		if(outcome == 'win') {
			let winner = game.getWinner()
			showNotification(winnerInfo(winner))
			states = game.getStates()
			reset()
		} else if(outcome == 'tie') {
			showNotification('Tie')
			states = game.getStates()
			reset()
		} else if(outcome == 'valid') {
			states = game.getStates()
		} else {
			showNotification('Impossible Move')
		}
  }
	
	/* display given text as a notification for 2 seconds */
	function showNotification(new_notification) {
		notification = new_notification
		show_notificaiton = true
		clearTimeout()
		setTimeout(function(){ show_notificaiton = false; }, 2000);
	}

	/* helper function -> format win message */
	function winnerInfo(winner){
		if(winner == 1) 
			return 'Player 1 Wins'
		else {
			if(game.agentPlaying()) 
				return 'Agent Wins'
			else
				return 'Player 2 Wins'
		}
	}

	/* Reset the game state */
	function reset(){
		clearTimeout()
		setTimeout(function(){ 
			game.reset(); 
			states=game.getStates() 
			turn = game.getTurn()
		}, 1500);
	}
</script>

<div 
	class="flex justify-center h-screen bg-black sm:items-center" 
	style="font-family: 'Bangers', cursive;">
	<div class="min-w-md"> 
		<!-- notification -->
		{#if show_notificaiton}
			<Notification>{notification}</Notification>
		{/if}
		<!-- title -->
		<h1 class="pt-10 text-6xl text-center text-white sm:pt-0">Tic-Tac-Toe</h1>
		<!-- turn -->
		<div class="py-10">
			<Turn 
				{turn}
				{agent_play}
			/>
		</div>
		<Board 
			states={states}
			{selectMarker}
		/>
		<!-- score -->
		<div class="py-10">
			<Score 
				{score}
				{agent_play}
			/>
		</div>
		<!-- reset -->
		<div class="flex justify-center pt-5">
			<span class="inline-flex rounded-md shadow-sm">
				<button on:click={()=>reset()} class="text-4xl text-white cursor-pointer focus:outline-none hover:text-gray-300">
					Reset
				</button>
			</span>
		</div>
	</div>
</div>
